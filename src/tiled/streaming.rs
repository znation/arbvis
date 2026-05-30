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
use crate::tiled::html::generate_leaflet_content;
use crate::tiled::leaf::{TileFormat, TILE};
use crate::tiled::pyramid_accum::{PyramidAccumulator, TileSink};

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
    let client = crate::hf_url::client()?;

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
    let tile_size = TILE;
    let max_zoom = plan.max_zoom;
    let world_w = plan.world_w;
    let world_h = plan.world_h;
    let height = plan.height;
    let width = plan.width;
    let total_tiles = plan.total_tiles;
    let leaf_ext = leaf_format.extension();
    let pyramid_ext = pyramid_format.extension();

    let sink = Arc::new(HfTileSink::new(client, hf_out.clone())?);
    let pyramid_path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync> = {
        let hf_out = hf_out.clone();
        let ext = pyramid_ext.to_string();
        Arc::new(move |z, x, y| hf_out.tile_repo_path(z, x, y, &ext))
    };
    let pyramid = Arc::new(PyramidAccumulator::new(
        tile_size,
        max_zoom,
        sink.clone(),
        pyramid_path_fn,
        pyramid_format,
    ));

    log::info!(
        "Rendering and uploading {} leaf tiles ({} leaf / {} pyramid)...",
        total_tiles,
        leaf_ext,
        pyramid_ext
    );

    let sink_for_write = sink.clone();
    let pyramid_for_write = pyramid.clone();
    let hf_out_for_write = hf_out.clone();
    drive_pipeline(
        &plan,
        leaf_format,
        max_zoom,
        TileCoords::Dense {
            width_tiles: plan.width_tiles,
            height_tiles: plan.height_tiles,
        },
        move |t: EncodedTile| {
            let repo_path = hf_out_for_write.tile_repo_path(max_zoom, t.tx, t.ty, leaf_ext);
            sink_for_write.upload_tile(repo_path, t.bytes)?;
            pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
            Ok(())
        },
    )
    .await?;

    // Await any in-flight pyramid encode/upload tasks before commit so every
    // staged file is on disk by the time hf-hub takes the snapshot.
    pyramid.drain().await;

    // Variable-depth detail tiles (sparse deeper levels, no accumulation).
    let detail_sink = sink.clone();
    let detail_hf_out = hf_out.clone();
    render_detail_levels(&plan, leaf_format, &move |t: &EncodedTile, z| {
        let repo_path = detail_hf_out.tile_repo_path(z, t.tx, t.ty, leaf_ext);
        detail_sink.upload_tile(repo_path, t.bytes.clone())
    })
    .await?;

    log::info!("Uploading index.html and labels.json...");
    let (html_bytes, labels_bytes) = generate_leaflet_content(
        world_w,
        world_h,
        max_zoom,
        plan.detail_depth,
        height,
        width,
        TILE,
        &plan.entities,
        title,
        inputs,
        leaf_ext,
        pyramid_ext,
    );
    sink.upload_tile(hf_out.index_html_path(), html_bytes.clone())?;
    sink.upload_tile(hf_out.labels_json_path(), labels_bytes)?;

    drop(pyramid);

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
