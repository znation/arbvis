//! Streaming tile output: render tiles and upload them directly to the Hub
//! without writing the pyramid to local disk.
//!
//! Off-by-default; gated behind the top-level `--stream` flag. Disk-backed
//! rendering (the `run_tiles` path) is the recommended mode for everything
//! that fits on disk because re-running an interrupted render doesn't cost a
//! re-download of every input file. We keep streaming around as an escape
//! hatch for inputs that don't fit on local disk; if that's ever solved by
//! something like hf-mount, this whole file can be deleted in one pass.

use std::sync::Arc;

use crate::hf_upload::HfTileSink;
use crate::hf_url::HfOutputSpec;
use crate::layout::LayoutMode;
use crate::tiled::html::{generate_leaflet_content, generate_leaflet_content_multi, SceneView};
use crate::tiled::leaf::{TileFormat, TILE};
use crate::tiled::pyramid_accum::{drain_and_report_incomplete, PyramidAccumulator, TileSink};

use super::scenes::{partition_scenes, SceneGroup};
use super::{
    build_tile_plan, derive_leaf_format, drive_pipeline, render_detail_levels, EncodedTile,
    TileCoords,
};
use crate::data::Source;

/// Run the tiled/pyramidal output pipeline, streaming tiles directly to HuggingFace Hub.
pub async fn run_tiles_hf_streaming(
    sources: Vec<Source>,
    total: u64,
    hf_out: &HfOutputSpec,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<Vec<u8>> {
    crate::hf_url::require_token()?;

    let scenes = partition_scenes(sources, total);
    let sink = Arc::new(HfTileSink::new(hf_out.clone())?);

    // Render + upload each scene's pyramid in turn; collect the per-scene
    // geometry for the (single- or multi-scene) viewer built at the end.
    let mut views: Vec<SceneView> = Vec::with_capacity(scenes.len());
    for group in scenes {
        let scene_key = group.key.clone();
        let view = stream_scene(
            &sink,
            hf_out,
            scene_key.as_deref(),
            group,
            diff_mode,
            show_xet_xorbs,
            leaf_format,
            pyramid_format,
            layout_mode,
            registry,
        )
        .await?;
        views.push(view);
    }

    log::info!("Uploading index.html and labels.json...");
    // The lone implicit scene takes the legacy single-layer viewer verbatim;
    // anything tagged gets the multi-scene tab switcher.
    let (html_bytes, labels_bytes) = if views.len() == 1 && views[0].key.is_none() {
        let v = &views[0];
        generate_leaflet_content(
            v.world_w,
            v.world_h,
            v.max_zoom,
            v.detail_depth,
            v.height,
            v.width,
            TILE,
            &v.entities,
            title,
            inputs,
            &v.leaf_ext,
            &v.pyramid_ext,
            &registry.branding,
        )
    } else {
        generate_leaflet_content_multi(&views, title, inputs, &registry.branding)
    };
    sink.upload_tile(hf_out.index_html_path(), html_bytes.clone())?;
    sink.upload_tile(hf_out.labels_json_path(), labels_bytes)?;

    log::info!("Creating HF Hub commit...");
    Arc::try_unwrap(sink)
        .map_err(|_| anyhow::anyhow!("unexpected extra Arc reference to tile sink"))?
        .commit("Add arbvis visualization tiles")
        .await?;

    log::info!(
        "Streaming output committed to hf://{}/{}",
        hf_out.repo_id,
        hf_out.path_prefix
    );
    Ok(html_bytes)
}

/// Render one scene's full pyramid (overview + detail) and stream it to the Hub
/// under `[<prefix>/]tiles/[<scene>/]…`, returning its [`SceneView`]. The
/// streaming analogue of `super::pipeline::render_scene_to_disk`; `scene = None` is the
/// legacy lone-pyramid layout. Shares `sink` with its siblings — every clone it
/// takes is dropped before it returns, so the caller can still `Arc::try_unwrap`.
/// Generic over the sink so tests can record uploads without touching the Hub.
#[allow(clippy::too_many_arguments)]
async fn stream_scene<S: TileSink>(
    sink: &Arc<S>,
    hf_out: &HfOutputSpec,
    scene: Option<&str>,
    group: SceneGroup,
    diff_mode: bool,
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<SceneView> {
    let SceneGroup {
        key,
        label,
        order,
        sources,
        total,
    } = group;

    let plan = build_tile_plan(
        sources,
        total,
        diff_mode,
        show_xet_xorbs,
        layout_mode,
        registry,
    )
    .await?;
    let leaf_format = derive_leaf_format(leaf_format, &plan.mode);
    let max_zoom = plan.max_zoom;
    let total_tiles = plan.total_tiles;
    let leaf_ext = leaf_format.extension();
    let pyramid_ext = pyramid_format.extension();

    let pyramid_path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync> = {
        let hf_out = hf_out.clone();
        let ext = pyramid_ext.to_string();
        let scene = scene.map(|s| s.to_string());
        Arc::new(move |z, x, y| hf_out.tile_repo_path_in(scene.as_deref(), z, x, y, &ext))
    };
    let pyramid = Arc::new(PyramidAccumulator::new(
        TILE,
        sink.clone(),
        pyramid_path_fn,
        pyramid_format,
    ));

    log::info!(
        "Rendering and uploading {} leaf tiles for {} ({} leaf / {} pyramid)...",
        total_tiles,
        scene.unwrap_or("tiles"),
        leaf_ext,
        pyramid_ext
    );

    let sink_for_write = sink.clone();
    let pyramid_for_write = pyramid.clone();
    let hf_out_for_write = hf_out.clone();
    let scene_for_write = scene.map(|s| s.to_string());
    drive_pipeline(
        &plan,
        leaf_format,
        max_zoom,
        TileCoords::Dense {
            width_tiles: plan.width_tiles,
            height_tiles: plan.height_tiles,
        },
        move |t: EncodedTile| {
            let repo_path = hf_out_for_write.tile_repo_path_in(
                scene_for_write.as_deref(),
                max_zoom,
                t.tx,
                t.ty,
                leaf_ext,
            );
            sink_for_write.upload_tile(repo_path, t.bytes)?;
            pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
            Ok(())
        },
    )
    .await?;

    // Await any in-flight pyramid encode/upload tasks before the next scene (or
    // commit) so every staged file is present when the folder is walked.
    drain_and_report_incomplete(pyramid).await?;

    // Variable-depth detail tiles (sparse deeper levels, no accumulation).
    let detail_sink = sink.clone();
    let detail_hf_out = hf_out.clone();
    let scene_for_detail = scene.map(|s| s.to_string());
    render_detail_levels(&plan, leaf_format, &move |t: &EncodedTile, z| {
        let repo_path =
            detail_hf_out.tile_repo_path_in(scene_for_detail.as_deref(), z, t.tx, t.ty, leaf_ext);
        detail_sink.upload_tile(repo_path, t.bytes.clone())
    })
    .await?;

    Ok(SceneView {
        key,
        label,
        order,
        world_w: plan.world_w,
        world_h: plan.world_h,
        max_zoom,
        detail_depth: plan.detail_depth,
        height: plan.height,
        width: plan.width,
        leaf_ext: leaf_ext.to_string(),
        pyramid_ext: pyramid_ext.to_string(),
        entities: plan.entities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Source, SourceKind};
    use std::sync::Mutex;

    /// Records every staged (repo_path, bytes) pair; the pyramid accumulator
    /// runs encode tasks on this sink, so a Mutex keeps it Sync-safe.
    #[derive(Default)]
    struct RecordingSink {
        uploads: Mutex<Vec<(String, Vec<u8>)>>,
    }

    impl TileSink for RecordingSink {
        fn upload_tile(&self, path: String, bytes: Vec<u8>) -> anyhow::Result<()> {
            self.uploads.lock().unwrap().push((path, bytes));
            Ok(())
        }
    }

    fn tiny_source() -> Source {
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(vec![0xa5u8; 64]),
            byte_size: 64,
            name_override: Some("tiny.bin".to_string()),
            xet_terms: None,
            extensions: Default::default(),
        }
    }

    fn spec() -> HfOutputSpec {
        HfOutputSpec {
            repo_id: "test/repo".to_string(),
            kind: crate::hf_url::RepoKind::Model,
            revision: "main".to_string(),
            path_prefix: "viz".to_string(),
        }
    }

    fn group() -> SceneGroup {
        SceneGroup {
            key: None,
            label: String::new(),
            order: 0,
            sources: vec![tiny_source()],
            total: 64,
        }
    }

    async fn stream_once(sink: &Arc<RecordingSink>) -> SceneView {
        stream_scene(
            sink,
            &spec(),
            None,
            group(),
            false,
            false,
            TileFormat::Png,
            TileFormat::Png,
            crate::layout::LayoutMode::Hilbert,
            &crate::registry::Registry::with_defaults(),
        )
        .await
        .unwrap()
    }

    /// 64 bytes → one Hilbert square → a single overview zoom level.
    #[tokio::test]
    async fn streams_leaf_and_pyramid_tiles_without_disk() {
        let sink = Arc::new(RecordingSink::default());
        let view = stream_once(&sink).await;

        assert_eq!(view.key, None);
        assert_eq!(view.leaf_ext, "png");
        assert_eq!(view.pyramid_ext, "png");
        assert_eq!(view.max_zoom, 0);
        assert_eq!(view.detail_depth, 0);
        assert!(view.world_w > 0 && view.world_h > 0);

        let uploads = sink.uploads.lock().unwrap();
        // The leaf tile and every pyramid tile it contributes are staged via
        // the same TileSink path the real HfTileSink would upload from.
        assert!(!uploads.is_empty(), "expected at least the leaf tile");
        let leaf: Vec<_> = uploads
            .iter()
            .filter(|(p, _)| p.starts_with("viz/tiles/0/"))
            .collect();
        assert_eq!(leaf.len(), 1, "one leaf at max_zoom 0: {uploads:?}");
        assert_eq!(leaf[0].0, "viz/tiles/0/0/0.png");
        assert!(!leaf[0].1.is_empty());
        assert!(uploads.iter().any(|(p, _)| *p == "viz/tiles/0/0/0.png"));
        // Every staged path is under the prefix + tiles/, none absolute.
        for (p, _) in uploads.iter() {
            assert!(p.starts_with("viz/tiles/"), "unexpected path {p}");
        }
    }

    /// A tagged scene routes its tiles under `tiles/<key>/` and carries the
    /// tag's label/order into the returned view.
    #[tokio::test]
    async fn tagged_scene_prefixes_tile_paths_and_carries_metadata() {
        use crate::data::SceneTag;

        let mut src = tiny_source();
        src.extensions.insert(SceneTag {
            key: "alpha".to_string(),
            label: "Alpha layer".to_string(),
            order: 3,
        });
        let group = SceneGroup {
            key: Some("alpha".to_string()),
            label: "Alpha layer".to_string(),
            order: 3,
            sources: vec![src],
            total: 64,
        };

        let sink = Arc::new(RecordingSink::default());
        let view = stream_scene(
            &sink,
            &spec(),
            Some("alpha"),
            group,
            false,
            false,
            TileFormat::Png,
            TileFormat::Png,
            crate::layout::LayoutMode::Hilbert,
            &crate::registry::Registry::with_defaults(),
        )
        .await
        .unwrap();

        assert_eq!(view.key.as_deref(), Some("alpha"));
        assert_eq!(view.label, "Alpha layer");
        assert_eq!(view.order, 3);
        let uploads = sink.uploads.lock().unwrap();
        assert!(
            uploads
                .iter()
                .all(|(p, _)| p.starts_with("viz/tiles/alpha/")),
            "all paths under the scene key: {uploads:?}"
        );
        assert!(uploads
            .iter()
            .any(|(p, _)| *p == "viz/tiles/alpha/0/0/0.png"));
    }
}
