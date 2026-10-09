//! Disk-backed tile rendering: `run_tiles` (the atomic replace/restore
//! wrapper), `run_tiles_inner`, and `render_scene_to_disk`. Split out of
//! `tiled/mod.rs` to keep each module readable in one sitting; the pipeline
//! machinery (`build_tile_plan`, `drive_pipeline`, `render_detail_levels`,
//! geometry constants) stays in the parent module and is reached via `super`.

use super::*;

/// Run the tiled/pyramidal output pipeline to a local directory.
///
/// `leaf_format` controls how each leaf tile is encoded; the actual format
/// may be upgraded to `IndexedPng` when the render mode produces ≤256-color
/// tiles (see [`derive_leaf_format`]). `pyramid_format` controls the
/// downsampled levels (default AVIF q≈85, since averaged pixels already
/// smudge the palette). The pyramid is built in memory by the streaming
/// accumulator as leaves complete, so we never re-decode tiles back from
/// disk — which is also why this code path doesn't need a decoder for AVIF
/// tiles.
pub async fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<()> {
    // Clear any tiles left over from a previous run before writing this one.
    // Output dirs are routinely reused (e.g. `-o tmp_out`), and a prior run on
    // a differently-sized input leaves a different pyramid shape behind — extra
    // zoom levels, a different leaf grid, even a different tile format (png vs
    // avif) at the same coords. None of that gets overwritten by this run, so it
    // lingers and can surface as black/garbage tiles at some zoom levels (e.g.
    // when a stale `index.html` with a higher `maxNativeZoom` is cached). The
    // whole tile tree is replaced so each run stays self-consistent; `index.html`
    // / `labels.json` are regenerated below regardless.
    //
    // The old tree is *renamed aside* rather than deleted up front: deleting
    // first means a run that fails halfway (bad input, full disk) leaves the
    // bundle without any tiles while `index.html` / `labels.json` still point
    // at the old pyramid — a working bundle destroyed by a failed re-render.
    // The rename is atomic, so even a process killed between rename and render
    // leaves the previous tiles recoverable as `tiles.stale-<pid>`; on success
    // the backup is deleted, on failure it is restored. Leftover backups from
    // runs killed mid-render are cleared at the start of each run (concurrent
    // renders into one directory were already unsupported: the old code raced
    // its `remove_dir_all` the same way).
    let tiles_root = tile_dir.join("tiles");
    remove_stale_tile_backups(&tile_dir)?;
    let trash = if tiles_root.exists() {
        let backup = tile_dir.join(format!("tiles.stale-{}", std::process::id()));
        std::fs::rename(&tiles_root, &backup)
            .with_context(|| format!("setting aside stale tiles in {}", tiles_root.display()))?;
        Some(backup)
    } else {
        None
    };

    let res = run_tiles_inner(
        sources,
        total,
        tile_dir.clone(),
        diff_mode,
        title,
        inputs,
        show_xet_xorbs,
        leaf_format,
        pyramid_format,
        layout_mode,
        registry,
    )
    .await;
    match res {
        Ok(()) => {
            if let Some(ref backup) = trash {
                if let Err(e) = std::fs::remove_dir_all(backup) {
                    log::warn!(
                        "could not remove stale tile backup {}: {e}",
                        backup.display()
                    );
                }
            }
            Ok(())
        }
        Err(err) => {
            if let Some(ref backup) = trash {
                if !tiles_root.exists() {
                    match std::fs::rename(backup, &tiles_root) {
                        Ok(()) => log::warn!(
                            "render failed; restored previous tiles from {}",
                            backup.display()
                        ),
                        Err(e) => log::warn!(
                            "render failed and previous tiles could not be restored \
                             from {} to {}: {e}",
                            backup.display(),
                            tiles_root.display()
                        ),
                    }
                }
            }
            Err(err)
        }
    }
}

/// Delete `tiles.stale-*` backup directories left in `tile_dir` by earlier
/// runs that were killed between setting aside the old tile tree and cleaning
/// it up.
fn remove_stale_tile_backups(tile_dir: &Path) -> anyhow::Result<()> {
    // A fresh render targets a tile_dir that does not exist yet; there is
    // nothing to sweep there.
    let entries = match std::fs::read_dir(tile_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("listing {}", tile_dir.display())),
    };
    for entry in entries {
        let entry = entry.with_context(|| format!("listing {}", tile_dir.display()))?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("tiles.stale-") {
            std::fs::remove_dir_all(entry.path()).with_context(|| {
                format!("clearing stale tile backup {}", entry.path().display())
            })?;
        }
    }
    Ok(())
}

/// Render the tile pyramid and viewer HTML into `tile_dir` (see [`run_tiles`]).
/// Assumes any stale `tiles/` tree has already been set aside by the caller.
async fn run_tiles_inner(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<()> {
    let scenes = partition_scenes(sources, total);
    let mut views: Vec<html::SceneView> = Vec::with_capacity(scenes.len());
    for group in scenes {
        let subdir = match &group.key {
            Some(k) => format!("tiles/{k}"),
            None => "tiles".to_string(),
        };
        let view = render_scene_to_disk(
            &tile_dir,
            &subdir,
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

    log::info!("Writing HTML viewer...");
    // The lone implicit scene takes the legacy single-layer viewer verbatim
    // (byte-identical output); anything tagged gets the multi-scene switcher.
    if views.len() == 1 && views[0].key.is_none() {
        let v = &views[0];
        html::write_leaflet_html(
            &tile_dir,
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
        )?;
    } else {
        html::write_leaflet_html_multi(&tile_dir, &views, title, inputs, &registry.branding)?;
    }

    log::info!("Tiled output written to {}", tile_dir.display());
    Ok(())
}

/// Render one scene's full pyramid (overview + detail levels) to disk under
/// `<tile_dir>/<subdir>/…` and return its [`html::SceneView`]. `subdir` is
/// `"tiles"` for the legacy lone scene or `"tiles/<key>"` for a named scene —
/// it's the only thing that distinguishes the two paths on disk.
#[allow(clippy::too_many_arguments)]
async fn render_scene_to_disk(
    tile_dir: &Path,
    subdir: &str,
    group: SceneGroup,
    diff_mode: bool,
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<html::SceneView> {
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

    std::fs::create_dir_all(tile_dir.join(format!("{subdir}/{max_zoom}")))?;

    log::info!(
        "Rendering {} leaf tiles for {} ({} leaf / {} pyramid)...",
        total_tiles,
        subdir,
        leaf_ext,
        pyramid_ext
    );

    let sink = Arc::new(LocalFileSink {
        root: tile_dir.to_path_buf(),
    });
    let pyramid_path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync> = {
        let ext = pyramid_ext.to_string();
        let subdir = subdir.to_string();
        Arc::new(move |z, x, y| format!("{subdir}/{z}/{x}/{y}.{ext}"))
    };
    let pyramid = Arc::new(PyramidAccumulator::new(
        TILE,
        sink.clone(),
        pyramid_path_fn,
        pyramid_format,
    ));

    let tile_dir_for_write = tile_dir.to_path_buf();
    let subdir_for_write = subdir.to_string();
    let pyramid_for_write = pyramid.clone();
    drive_pipeline(
        &plan,
        leaf_format,
        max_zoom,
        TileCoords::Dense {
            width_tiles: plan.width_tiles,
            height_tiles: plan.height_tiles,
        },
        move |t: EncodedTile| {
            let path = tile_dir_for_write.join(format!(
                "{subdir_for_write}/{max_zoom}/{}/{}.{leaf_ext}",
                t.tx, t.ty
            ));
            write_tile_file(&path, &t.bytes)?;
            pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
            Ok(())
        },
    )
    .await?;

    log::info!("Draining pyramid encode tasks...");
    pyramid
        .drain()
        .await
        .context("pyramid overview-tile encode/upload failed; the tile set is incomplete")?;
    drop(pyramid);

    // Variable-depth detail tiles: sparse deeper levels rendered directly from
    // source over the shrunk tensors' footprints (no pyramid accumulation).
    let detail_dir = tile_dir.to_path_buf();
    let subdir_for_detail = subdir.to_string();
    render_detail_levels(&plan, leaf_format, &move |t: &EncodedTile, z| {
        let path = detail_dir.join(format!(
            "{subdir_for_detail}/{z}/{}/{}.{leaf_ext}",
            t.tx, t.ty
        ));
        write_tile_file(&path, &t.bytes)?;
        Ok(())
    })
    .await?;

    Ok(html::SceneView {
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
    use crate::data::{Source, SourceKind};

    fn file_source(path: &std::path::Path, len: u64) -> Source {
        Source {
            file_idx: 0,
            kind: SourceKind::File(path.to_path_buf()),
            byte_size: len,
            name_override: None,
            xet_terms: None,
            extensions: Default::default(),
        }
    }

    /// A failed re-render into an existing bundle must not destroy the old
    /// pyramid: the stale tree is set aside by rename and restored on error.
    #[tokio::test]
    async fn failed_run_tiles_restores_previous_tiles() {
        let dir = tempfile::tempdir().unwrap();
        let leaf = dir.path().join("tiles/0/0/0.png");
        std::fs::create_dir_all(leaf.parent().unwrap()).unwrap();
        std::fs::write(&leaf, b"OLD").unwrap();

        let sources = vec![file_source(&dir.path().join("does-not-exist.bin"), 64)];
        let err = crate::tiled::run_tiles(
            sources,
            64,
            dir.path().to_path_buf(),
            false,
            "t",
            &[],
            false,
            crate::tiled::TileFormat::IndexedPng,
            crate::tiled::TileFormat::IndexedPng,
            crate::layout::LayoutMode::Auto,
            &crate::registry::Registry::with_defaults(),
        )
        .await;
        assert!(err.is_err(), "expected the render to fail");
        assert_eq!(
            std::fs::read(&leaf).unwrap(),
            b"OLD",
            "old tiles must survive a failed re-render"
        );
        assert!(dir.path().read_dir().unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("tiles.stale-")));
    }

    /// A successful re-render into an existing bundle replaces the old tree
    /// and leaves no backup directory behind.
    #[tokio::test]
    async fn successful_run_tiles_replaces_previous_tiles_without_backups() {
        let dir = tempfile::tempdir().unwrap();
        let leaf = dir.path().join("tiles/0/0/0.png");
        std::fs::create_dir_all(leaf.parent().unwrap()).unwrap();
        std::fs::write(&leaf, b"OLD").unwrap();

        let mut bytes = vec![0u8; 64];
        bytes.extend(std::iter::repeat_n(0x41u8, 64));
        let keep = dir.path().join("input.bin");
        std::fs::write(&keep, &bytes).unwrap();
        let len = bytes.len() as u64;
        crate::tiled::run_tiles(
            vec![file_source(&keep, len)],
            len,
            dir.path().to_path_buf(),
            false,
            "t",
            &[],
            false,
            crate::tiled::TileFormat::IndexedPng,
            crate::tiled::TileFormat::IndexedPng,
            crate::layout::LayoutMode::Auto,
            &crate::registry::Registry::with_defaults(),
        )
        .await
        .unwrap();
        assert!(
            std::fs::read(&leaf).unwrap() != b"OLD",
            "tiles must be replaced"
        );
        assert!(dir.path().read_dir().unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("tiles.stale-")));
        assert!(dir.path().join("index.html").is_file());
    }

    /// A fresh render into a tile_dir that does not exist yet must succeed:
    /// the stale-backup sweep treats a missing directory as nothing to do.
    #[tokio::test]
    async fn fresh_run_tiles_into_missing_directory_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = vec![0u8; 64];
        bytes.extend(std::iter::repeat_n(0x41u8, 64));
        let keep = dir.path().join("input.bin");
        std::fs::write(&keep, &bytes).unwrap();
        let len = bytes.len() as u64;
        crate::tiled::run_tiles(
            vec![file_source(&keep, len)],
            len,
            dir.path().join("out").to_path_buf(),
            false,
            "t",
            &[],
            false,
            crate::tiled::TileFormat::IndexedPng,
            crate::tiled::TileFormat::IndexedPng,
            crate::layout::LayoutMode::Auto,
            &crate::registry::Registry::with_defaults(),
        )
        .await
        .unwrap();
        assert!(dir.path().join("out/index.html").is_file());
    }
}
