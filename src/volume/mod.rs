//! 3D (`--3d`) render path: aggregate the source bytes onto a 3D Hilbert curve
//! inside a cube and emit a self-contained Three.js viewer bundle.
//!
//! This is the 3D analog of [`crate::tiled`]. Where the 2D path lays one pixel
//! per byte and builds a tile pyramid, the 3D path lays bytes along a 3D
//! Hilbert curve and aggregates them into a bounded voxel grid — so render and
//! download cost are governed by the grid resolution, not the (potentially
//! many-GB) input size. The viewer ray-marches the grid with opacity encoding
//! density (so the cube's interior is visible).
//!
//! Output bundle (written to the `--out` directory, deployed verbatim by
//! `--space`): `index.html`, `volume.bin` (RGBA8 `Data3DTexture` payload),
//! `bricks.bin` / `pagetable.bin` (the sparse brick pool the ray-march reads),
//! and `meta.json`.

mod aggregate;
pub mod brick;
mod brick_stream;
pub mod encode;
pub mod html;
pub mod shape;
pub mod voxel;

pub use shape::{
    select_volume_shape, HilbertVolumePlugin, VolumeEntity, VolumeLabel, VolumeShape, VoxelBox,
};
pub use voxel::{VoxelCell, VoxelGridMut, VoxelRegistry, VoxelRenderCtx, VoxelRenderer};

use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::Source;
use crate::layout::LayoutMode;
use crate::registry::{Branding, Registry};
use aggregate::{aggregate_bytes_hilbert, aggregate_entities};
use encode::VolumeMeta;

pub(crate) use aggregate::box_focus;

pub(crate) use crate::fsutil::{part_path, seal_part};

/// Write `bytes` to `path` atomically: stage to `<file>.part` in the same
/// directory, then rename over `path`. A process killed mid-write leaves the
/// previous file — or none — instead of a truncated artifact: the viewer
/// bundle is read back by `regen_html` and served verbatim by the deployed
/// Space, so a half-written `meta.json` or `bricks.bin` would be served as
/// if complete.
fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let part = part_path(path);
    std::fs::write(&part, bytes)
        .with_context(|| format!("writing {}", path.display()))
        .inspect_err(|_| {
            // Best effort: don't leave a stale partial staging file behind — a
            // later `hf upload` of the bundle directory would push it to the
            // Hub, and it is indistinguishable from an in-progress staging file.
            let _ = std::fs::remove_file(&part);
        })?;
    seal_part(&part, path)
}
/// The dense `volume.bin` (coarse fallback LOD + CPU pick/histogram buffer) is
/// capped at this side so the mandatory up-front download stays small and fixed
/// (256³·4 ≈ 64 MiB for a full cube; far smaller for the thin, aspect-preserving
/// boxes structured layouts produce) regardless of the requested detail
/// resolution. Detail finer than this streams on demand from the sparse brick
/// pool. Sized so the always-resident fallback is already reasonably sharp — a
/// stall or slow stream degrades to a legible 256³ image, not an 128³ blur.
pub(crate) const COARSE_CAP: u32 = 256;

/// Peak slab-buffer budget for the streamed structured path. The driver picks a
/// brick-aligned `slab_depth` so `ex·ey·slab_depth·size_of::<VoxelCell>()` stays
/// under this — bounding peak RAM to one slab (plus the O(occupied) octree and
/// the bounded coarse accumulator) instead of the full `extent³` dense grid.
const SLAB_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// Split the requested detail resolution (`--grid`) into the effective
/// `(coarse dense-grid side, streamed brick resolution)` the byte volume path
/// consumes:
///
/// * An explicit `--volume-res` is an advanced override — keep the historical
///   meaning: coarse grid at `--grid`, streamed pool at `--volume-res`.
/// * Otherwise, anything finer than [`COARSE_CAP`] is streamed: build the coarse
///   dense grid at `COARSE_CAP` and the brick pool at the full `--grid`, so the
///   up-front download is bounded while detail arrives on demand.
/// * At or below the cap, keep the simple dense path (no streaming, `0`).
///
/// Only the byte path streams (the brick pool is byte-only); structured layouts
/// bypass this and keep the full dense grid — see [`render_volume`].
pub(crate) fn derive_volume_resolution(grid: u32, volume_res: u32) -> (u32, u32) {
    if volume_res != 0 {
        (grid, volume_res)
    } else if grid > COARSE_CAP {
        (COARSE_CAP, grid)
    } else {
        (grid, 0)
    }
}

/// Everything `render_volume` needs to emit the 3D viewer bundle.
pub(crate) struct BuildResult {
    /// Dense (or slab-folded) RGBA volume as written to `volume.bin`.
    pub(crate) volume_rgba: Vec<u8>,
    /// Extent of the grid actually written to `volume.bin` (`volume_rgba`). Equals
    /// the bake extent for the dense paths, but the **coarse** extent when the
    /// structured path streams (the full detail lives in `bricks`, and `volume.bin`
    /// is a small aspect-preserving downsample for the fallback LOD + CPU pick).
    pub(crate) grid_extent: [u32; 3],
    /// Largest per-voxel byte count; copied into `VolumeMeta::max_count`.
    pub(crate) max_count: u64,
    /// Camera framing of the occupied region; copied into the meta fields of
    /// the same names.
    pub(crate) focus_center: [f32; 3],
    /// Half-extent (world-space radius) of the occupied region.
    pub(crate) focus_radius: f32,
    /// Sparse brick pool built at a higher virtual resolution (byte floor with
    /// `--volume-res`, or the streamed structured path); `None` ⇒ `render_volume`
    /// derives bricks from the dense grid instead.
    pub(crate) bricks: Option<brick::BrickVolume>,
}

/// Render the 3D viewer bundle for `sources` into `out_dir`.
///
/// Picks a [`VolumeShape`] from the registry (mirroring 2D layout selection):
/// the `i32::MIN` [`HilbertVolumePlugin`] floor runs the legacy whole-stream
/// byte→Hilbert fill, while a higher-priority structured shape (e.g.
/// modelweightvis's `"arch"`) places per-tensor entities and colors them via a
/// [`VoxelRenderer`] that bakes final RGB into each voxel. The viewer's
/// `color_mode` (`"lut"` for the byte path, `"rgb"` for structured) is recorded
/// in `meta.json`.
#[allow(clippy::too_many_arguments)]
pub async fn render_volume(
    sources: Vec<Source>,
    total: u64,
    out_dir: PathBuf,
    title: &str,
    inputs: &[String],
    diff_mode: bool,
    grid_side: u32,
    volume_res: u32,
    mode: LayoutMode,
    registry: &Registry,
    branding: &Branding,
) -> anyhow::Result<()> {
    let pixel_lut = if diff_mode {
        build_diff_signed_lut()
    } else {
        build_pixel_lut()
    };

    // Pick the volume layout. Byte offsets are per-source for the entity path,
    // but `select_volume_shape` (and a downstream's `build`) still wants the
    // cumulative offsets, exactly like `select_layout`.
    let cumulative_offsets = cumulative_offsets(&sources);
    let shape = select_volume_shape(
        &sources,
        &cumulative_offsets,
        total,
        mode,
        diff_mode,
        grid_side,
        registry,
    )?;
    let is_byte = shape.is_byte_volume();
    // Byte volumes scale by streaming fine detail from the sparse brick pool
    // while keeping the dense grid (coarse fallback LOD + CPU pick) small and
    // fixed; `derive_volume_resolution` caps the coarse side and routes the
    // full requested resolution into the streamed pool. Structured layouts
    // don't stream (the pool is byte-only), so they keep the full dense grid the
    // shape sized. `volume_res` below is the effective streamed brick side.
    let (dense_side, volume_res) = if is_byte {
        derive_volume_resolution(grid_side, volume_res)
    } else {
        (grid_side, volume_res)
    };
    // The byte floor is a cube; override its extent to the (possibly capped)
    // coarse side so `volume.bin` stays small while detail streams.
    let actual_extent = if is_byte {
        [dense_side; 3]
    } else {
        shape.grid_extent()
    };
    let [ex, ey, ez] = actual_extent;
    let color_mode = if is_byte { "lut" } else { "rgb" };
    // Structured layouts above the coarse cap now stream too (like the byte path):
    // the full dense grid is baked, then diced into a sparse octree + range-served
    // brick pool while `volume.bin` ships as a small aspect-preserving coarse
    // downsample. Below the cap they keep the simple dense (non-streamed) path.
    let structured_streamed = !is_byte && actual_extent.iter().copied().max().unwrap() > COARSE_CAP;
    // Pick manifest for the click-to-pick viewer (empty for the byte floor).
    // Captured before `shape` moves into the blocking closure. Bboxes stay in the
    // full `vol_dim` voxel space; the viewer maps them onto the coarse grid.
    let manifest = shape.manifest();

    if is_byte {
        log::info!("Aggregating {total} bytes into a {ex}³ voxel grid via 3D Hilbert curve...");
    } else {
        log::info!(
            "Rendering structured `{}` 3D volume into a {ex}×{ey}×{ez} voxel box{}...",
            shape.id(),
            if structured_streamed {
                " (streamed)"
            } else {
                ""
            }
        );
    }

    // Create the output dir up front: the streamed builders write `bricks.bin`
    // incrementally (one brick at a time) from inside the blocking closure, so
    // the directory must exist before they run.
    std::fs::create_dir_all(&out_dir)?;

    // CPU + per-chunk fetch work on the blocking pool, like the single-image
    // path — keeps the tokio runtime free for the `Http`/`Xet`/`LazyDiff`
    // fetches the workers drive via `block_on`. The voxel registry is a cheap
    // Arc-map clone so the blocking closure owns everything it needs.
    let rt = tokio::runtime::Handle::current();
    let voxel_reg = registry.voxel.clone();
    let out_dir_build = out_dir.clone();
    let built = tokio::task::spawn_blocking(move || {
        if is_byte {
            // Byte volumes are cubes (Hilbert needs equal sides); all three
            // axes match, so the cube side is `ex`.
            aggregate_bytes_hilbert(
                sources,
                total,
                ex,
                volume_res,
                pixel_lut,
                rt,
                &out_dir_build,
            )
        } else {
            aggregate_entities(
                sources,
                shape.as_ref(),
                actual_extent,
                &voxel_reg,
                diff_mode,
                structured_streamed,
                rt,
                &out_dir_build,
            )
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("volume aggregation join failure: {e}"))??;

    write_atomic(&out_dir.join("volume.bin"), &built.volume_rgba)?;

    // Sparse brick pool + page table the volume ray-march renders from
    // (GigaVoxels-style indirection — only occupied bricks, empty ones leapt).
    // The dense volume.bin above stays for CPU-side histograms + pick.
    // Prefer the high-res streamed pool (--volume-res) when present; otherwise
    // derive bricks from the dense grid.
    let bricks = match built.bricks {
        Some(b) => b,
        // Non-streamed: derive bricks from the dense grid actually written
        // (`grid_extent` == the bake extent here, since streaming set `bricks`).
        None => brick::build_brick_volume(&built.volume_rgba, built.grid_extent, brick::BRICK),
    };
    // Streamed builders already wrote `bricks.bin` incrementally as bricks
    // finalized (their `atlas` is empty); only the non-streamed dense-derived
    // path still holds the atlas in RAM and writes it here.
    if !bricks.streamed {
        write_atomic(&out_dir.join("bricks.bin"), &bricks.atlas)?;
    }
    // Page structure: the streamed path ships a sparse octree node pool
    // (`tree.bin`); the non-streamed/flat path ships the dense page table
    // (`pagetable.bin`).
    let (page_file, tree_file) = if bricks.streamed {
        write_atomic(&out_dir.join("tree.bin"), &bricks.node_pool)?;
        (String::new(), "tree.bin".to_string())
    } else {
        write_atomic(&out_dir.join("pagetable.bin"), &bricks.page_table)?;
        ("pagetable.bin".to_string(), String::new())
    };

    // Report the blocking up-front download: the coarse dense grid + the sparse
    // octree node pool (bricks.bin streams on demand and is excluded). The octree
    // is O(occupied), not O((side/BRICK)³), so this stays small however high the
    // detail resolution; the deployed Space gzips both assets on the wire (see
    // space_template/app.py.tmpl).
    if bricks.streamed {
        let upfront = built.volume_rgba.len() + bricks.node_pool.len();
        let [cx, cy, cz] = built.grid_extent;
        // bricks.bin was streamed to disk (not held in RAM); its size is the
        // occupied-block count × brick bytes (flat blocks, no apron).
        let bricks_bytes = bricks.occupied as u64 * (brick::BRICK as u64).pow(3) * 4;
        log::info!(
            "3D up-front download ≈ {:.1} MiB (coarse {cx}×{cy}×{cz} grid {:.1} MiB + octree {:.1} MiB, \
             {} nodes, depth {}); {} bricks stream on demand from bricks.bin ({:.1} MiB)",
            upfront as f64 / (1 << 20) as f64,
            built.volume_rgba.len() as f64 / (1 << 20) as f64,
            bricks.node_pool.len() as f64 / (1 << 20) as f64,
            bricks.node_count,
            bricks.tree_depth,
            bricks.occupied,
            bricks_bytes as f64 / (1 << 20) as f64,
        );
    }
    let brick_meta = encode::BrickVolumeMeta {
        atlas_file: "bricks.bin".to_string(),
        page_file,
        brick: brick::BRICK,
        page_dim: bricks.page_dim,
        atlas_dim: bricks.atlas_dim,
        vol_dim: bricks.vol_dim,
        apron: bricks.apron,
        occupied: bricks.occupied,
        streamed: bricks.streamed,
        tree_file,
        tree_dim: bricks.node_pool_dim,
        tree_depth: bricks.tree_depth,
        node_count: bricks.node_count,
        max_count: bricks.max_count,
    };

    let meta = VolumeMeta {
        title: title.to_string(),
        brand_name: branding.name.to_string(),
        repo_url: branding.repo_url.to_string(),
        grid_extent: built.grid_extent,
        total_bytes: total,
        max_count: built.max_count,
        diff_mode,
        color_mode: color_mode.to_string(),
        inputs: inputs.to_vec(),
        focus_center: built.focus_center,
        focus_radius: built.focus_radius,
        lut: pixel_lut.iter().map(|c| c.0).collect(),
        manifest,
        // v5: the streamed path's page structure is a sparse octree node pool
        // (`bricks.tree_*`) rather than a flat page table. v6: the streamed byte
        // atlas ships RAW density counts and the viewer normalizes by
        // `bricks.max_count` in-shader (deferred so bricks stream to disk).
        format_version: 6,
        bricks: Some(brick_meta),
    };
    write_atomic(&out_dir.join("meta.json"), &serde_json::to_vec(&meta)?)?;
    write_atomic(
        &out_dir.join("index.html"),
        html::build_volume_html(title, inputs, branding).as_bytes(),
    )?;

    log::info!("3D viewer bundle written to {}", out_dir.display());
    Ok(())
}

/// Prefix-sum of source byte sizes: `out[i]` is the absolute start offset of
/// `sources[i]` in the concatenated stream. Matches the 2D tile path's
/// `cumulative_offsets`, so a downstream's byte arithmetic carries over.
fn cumulative_offsets(sources: &[Source]) -> Vec<u64> {
    let mut offs = Vec::with_capacity(sources.len());
    let mut acc = 0u64;
    for s in sources {
        offs.push(acc);
        acc += s.byte_size;
    }
    offs
}

/// Rebuild `index.html` for an existing 3D bundle from its `meta.json`,
/// Regenerate `index.html` for an existing 3D bundle directory without
/// re-aggregating. Mirrors [`crate::tiled::regen_html`]. An explicit
/// `--title` overrides the title persisted in `meta.json`, which otherwise
/// wins over the branding default.
pub fn regen_html(dir: &Path, branding: &Branding, title: Option<&str>) -> anyhow::Result<()> {
    let meta_path = dir.join("meta.json");
    let bytes = std::fs::read(&meta_path)
        .with_context(|| format!("reading {} (is this a --3d bundle?)", meta_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let title = title.map(String::from).unwrap_or_else(|| {
        v.get("title")
            .and_then(|t| t.as_str())
            .map(String::from)
            .unwrap_or_else(|| branding.name.to_string())
    });
    let inputs: Vec<String> = v
        .get("inputs")
        .and_then(|i| i.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let html = html::build_volume_html(&title, &inputs, branding);
    write_index_html_atomic(dir, html.as_bytes())?;
    log::info!("Regenerated 3D viewer index.html in {}", dir.display());
    Ok(())
}

/// Write the 3D viewer's `index.html` into `dir` via [`write_atomic`] (a
/// `.part` sibling renamed into place, so a process killed mid-write or an
/// ENOSPC partway through leaves the previous complete file instead of a
/// truncated one that the deployed viewer serves as if complete). If the
/// target path cannot be replaced (it exists as a directory), the write fails
/// before anything is staged, leaving the previous file untouched. Mirrors the
/// 2D viewer pair write in `crate::tiled::html::write_viewer_pair`.
fn write_index_html_atomic(dir: &Path, html: &[u8]) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating viewer dir {}", dir.display()))?;
    let index = dir.join("index.html");
    if index.is_dir() {
        anyhow::bail!(
            "cannot write viewer artifact {}: path exists as a directory",
            index.display()
        );
    }
    write_atomic(&index, html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SourceKind;
    use std::sync::Arc;

    #[test]
    fn derive_split_streams_detail_above_cap() {
        // A single high --grid becomes (coarse cap, streamed detail).
        assert_eq!(derive_volume_resolution(512, 0), (COARSE_CAP, 512));
        assert_eq!(derive_volume_resolution(2048, 0), (COARSE_CAP, 2048));
    }

    #[test]
    fn derive_split_stays_dense_at_or_below_cap() {
        // No streaming (0) at or below the cap; coarse grid == --grid.
        assert_eq!(derive_volume_resolution(COARSE_CAP, 0), (COARSE_CAP, 0));
        assert_eq!(derive_volume_resolution(64, 0), (64, 0));
    }

    #[test]
    fn derive_split_explicit_volume_res_overrides() {
        // The advanced knob keeps the historical meaning: coarse at --grid,
        // stream at --volume-res.
        assert_eq!(derive_volume_resolution(256, 1024), (256, 1024));
    }

    fn buffered(bytes: Vec<u8>) -> Source {
        let len = bytes.len() as u64;
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(bytes),
            byte_size: len,
            name_override: None,
            xet_terms: None,
            extensions: Default::default(),
        }
    }

    /// A half-zero / half-0xFF buffer must produce both fully-transparent
    /// (activity 0) and fully-active (activity 255) occupied voxels, and a
    /// well-formed bundle (volume.bin sized to the grid).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn aggregates_zero_and_ff_split() {
        let mut bytes = vec![0u8; 4096];
        bytes.extend(std::iter::repeat_n(0xFFu8, 4096));
        let total = bytes.len() as u64;
        let dir = tempfile::tempdir().unwrap();
        let grid_side = 32u32;
        render_volume(
            vec![buffered(bytes)],
            total,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            grid_side,
            0,
            LayoutMode::Auto,
            &Registry::with_defaults(),
            &Branding::default(),
        )
        .await
        .unwrap();

        let vol = std::fs::read(dir.path().join("volume.bin")).unwrap();
        assert_eq!(vol.len() as u64, (grid_side as u64).pow(3) * 4);

        let (mut zero_act, mut full_act, mut occupied) = (false, false, 0u64);
        for px in vol.chunks_exact(4) {
            if px[3] > 0 {
                occupied += 1;
                if px[1] == 0 {
                    zero_act = true; // the 0x00 half → transparent
                }
                if px[1] >= 250 {
                    full_act = true; // the 0xFF half → opaque
                }
            }
        }
        assert_eq!(occupied, total, "one byte per voxel below grid capacity");
        assert!(zero_act && full_act, "expected an activity split");

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(
            meta["grid_extent"],
            serde_json::json!([grid_side, grid_side, grid_side]),
            "byte floor is a cube"
        );
        assert_eq!(
            meta["color_mode"], "lut",
            "byte floor must stay LUT-colored"
        );
        assert!(dir.path().join("index.html").exists());
        // Atomic writes leave no staged .part files behind.
        for name in ["volume.bin", "bricks.bin", "meta.json", "index.html"] {
            assert!(
                !dir.path().join(format!("{name}.part")).exists(),
                "no staged {name}.part should remain"
            );
        }
    }

    /// A failure while regenerating the 3D viewer's index.html (here: the
    /// target exists as a directory, so it cannot be replaced) must fail
    /// loudly without leaving staging residue.
    #[test]
    fn regen_html_failure_leaves_no_part_residue() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("meta.json"), b"{}").unwrap();
        std::fs::create_dir(dir.path().join("index.html")).unwrap();
        let res = regen_html(dir.path(), &Branding::default(), None);
        assert!(res.is_err(), "write into a directory path must fail loudly");
        assert!(!dir.path().join("index.html.part").exists());
        // The existing directory is untouched by the failed regeneration.
        assert!(dir.path().join("index.html").is_dir());
    }

    /// A successful regen_html writes a complete index.html atomically: the
    /// .part sibling is gone and the file renders from meta.json inputs.
    #[test]
    fn regen_html_writes_index_atomically() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("meta.json"),
            serde_json::json!({"title": "T", "inputs": ["a.bin"]}).to_string(),
        )
        .unwrap();
        regen_html(dir.path(), &Branding::default(), None).unwrap();
        let html = std::fs::read(dir.path().join("index.html")).unwrap();
        assert!(!html.is_empty());
        assert!(!dir.path().join("index.html.part").exists());
    }

    /// `--title` must reach the regenerated HTML, overriding both the title
    /// persisted in `meta.json` and the branding default.
    #[test]
    fn regen_html_title_override_reaches_index_html() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("meta.json"),
            serde_json::json!({"title": "persisted", "inputs": []}).to_string(),
        )
        .unwrap();
        regen_html(dir.path(), &Branding::default(), Some("override")).unwrap();
        let html = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        assert!(
            html.contains("override"),
            "--title override must reach the HTML"
        );
        assert!(
            !html.contains("persisted"),
            "persisted title must be replaced"
        );

        // Without an override the persisted title still wins over branding.
        regen_html(dir.path(), &Branding::default(), None).unwrap();
        let html = std::fs::read_to_string(dir.path().join("index.html")).unwrap();
        assert!(
            html.contains("persisted"),
            "meta.json title must survive a bare regen"
        );
    }

    /// write_atomic must leave the previous file intact when the staged write
    /// fails, and leave no `.part` behind when it succeeds.
    #[test]
    fn write_atomic_replaces_and_fails_clean() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.bin");
        write_atomic(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        write_atomic(&path, b"second- longer").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second- longer");
        assert!(!dir.path().join("artifact.bin.part").exists());

        // Make the staged write fail (read-only dir) and check the previous
        // file survives untouched.
        use std::os::unix::fs::PermissionsExt;
        let ro = tempfile::tempdir().unwrap();
        let ro_dir = ro.path().join("ro");
        std::fs::create_dir(&ro_dir).unwrap();
        let keep_path = ro_dir.join("keep.bin");
        std::fs::write(&keep_path, b"original").unwrap();
        let mut dp = std::fs::metadata(&ro_dir).unwrap().permissions();
        dp.set_mode(0o500);
        std::fs::set_permissions(&ro_dir, dp).unwrap();
        assert!(write_atomic(&keep_path, b"new").is_err());
        assert_eq!(std::fs::read(&keep_path).unwrap(), b"original");
        assert!(!ro_dir.join("keep.bin.part").exists());
        let mut restore = std::fs::metadata(&ro_dir).unwrap().permissions();
        restore.set_mode(0o755);
        std::fs::set_permissions(&ro_dir, restore).unwrap();
        drop(ro);

        // Make the rename fail (target path exists as a directory) and check
        // the staged `.part` is cleaned up instead of lingering in the bundle
        // directory with the full artifact's bytes.
        let dir2 = tempfile::tempdir().unwrap();
        let clash = dir2.path().join("clash.bin");
        std::fs::create_dir(&clash).unwrap();
        assert!(write_atomic(&clash, b"payload").is_err());
        assert!(!dir2.path().join("clash.bin.part").exists());
    }

    /// A streamed brick-pool build must seal `bricks.bin` via rename, leaving
    /// no `.part` staged file.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streamed_bricks_bin_leaves_no_part_file() {
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i * 17 + 3) as u8).collect();
        let dir = tempfile::tempdir().unwrap();
        render_volume(
            vec![buffered(bytes)],
            200_000,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            8,
            16, // --volume-res > --grid ⇒ the streamed brick builder runs
            LayoutMode::Auto,
            &Registry::with_defaults(),
            &Branding::default(),
        )
        .await
        .unwrap();
        assert!(dir.path().join("bricks.bin").exists());
        assert!(
            !dir.path().join("bricks.bin.part").exists(),
            "sealed bricks.bin must not leave a .part staged file"
        );
    }

    /// `--volume-res` above `--grid` must emit the ray-guided **streamed** brick
    /// pool: `meta.bricks.streamed == true`, the page table holds 1-based brick
    /// ids, and `bricks.bin` is exactly `occupied · brick³ · 4` bytes (a flat,
    /// range-addressable block array the viewer streams into a bounded cache).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn emits_streamed_brick_pool_above_grid() {
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i * 31 + 7) as u8).collect();
        let total = bytes.len() as u64;
        let dir = tempfile::tempdir().unwrap();
        render_volume(
            vec![buffered(bytes)],
            total,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            8,  // --grid: dense voxel grid
            16, // --volume-res > --grid ⇒ the streaming brick builder runs
            LayoutMode::Auto,
            &Registry::with_defaults(),
            &Branding::default(),
        )
        .await
        .unwrap();

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        let bm = &meta["bricks"];
        assert_eq!(bm["streamed"], true, "--volume-res ships the streamed pool");
        assert_eq!(
            bm["vol_dim"][0], 16,
            "octree sized to the virtual resolution"
        );
        assert_eq!(bm["tree_depth"], 1, "depth = log2(16 / brick=8) = 1");
        let occupied = bm["occupied"].as_u64().unwrap();
        assert!(occupied > 0, "some bricks are occupied");

        // bricks.bin is a flat array of occupied blocks; the sparse octree node
        // pool (tree.bin) indexes them — no flat pagetable.bin in the streamed path.
        let bricks = std::fs::read(dir.path().join("bricks.bin")).unwrap();
        let brick = bm["brick"].as_u64().unwrap() as usize;
        assert_eq!(
            bricks.len() as u64,
            occupied * (brick.pow(3) * 4) as u64,
            "bricks.bin holds occupied flat brick blocks"
        );
        assert!(
            !dir.path().join("pagetable.bin").exists(),
            "streamed path emits no flat page table"
        );
        let tree = std::fs::read(dir.path().join("tree.bin")).unwrap();
        let td = &bm["tree_dim"];
        let texels = td[0].as_u64().unwrap() * td[1].as_u64().unwrap() * td[2].as_u64().unwrap();
        assert_eq!(
            tree.len() as u64,
            texels * 4,
            "node pool is tree_dim texels of RGBA8"
        );
        // Exactly one leaf entry (A>0) per occupied brick; its RGB brick id runs
        // 1..=occupied and addresses a real block in bricks.bin.
        let (mut leaves, mut max_id) = (0u64, 0u32);
        for e in tree.chunks_exact(4) {
            if e[3] > 0 {
                leaves += 1;
                max_id = max_id.max(e[0] as u32 | (e[1] as u32) << 8 | (e[2] as u32) << 16);
            }
        }
        assert_eq!(leaves, occupied, "one octree leaf per occupied brick");
        assert_eq!(max_id as u64, occupied, "leaf brick ids run 1..=occupied");
        // v6: the streamed byte atlas ships RAW density counts + a positive
        // max_count so the viewer normalizes density in-shader.
        assert!(
            bm["max_count"].as_u64().unwrap() > 0,
            "streamed byte atlas carries a positive max_count for shader-side density"
        );
    }

    // A structured VolumeShape whose single entity paints a 4³ box; its
    // VoxelRenderer bakes a fixed RGBA into the cube. Exercises the entity path
    // end to end: shape selection (priority over the floor), per-entity fetch,
    // voxel-renderer dispatch, baked-RGB packing, and the `"rgb"` color_mode in
    // `meta.json`.
    struct TestVolume {
        extent: [u32; 3],
    }
    impl VolumeShape for TestVolume {
        fn id(&self) -> &'static str {
            "test-vol"
        }
        fn grid_extent(&self) -> [u32; 3] {
            self.extent
        }
        fn entities(&self) -> Option<Vec<VolumeEntity>> {
            Some(vec![VolumeEntity {
                source_idx: 0,
                byte_start: 0,
                byte_len: 16,
                bbox: VoxelBox {
                    x0: 0,
                    y0: 0,
                    z0: 0,
                    x1: 4,
                    y1: 4,
                    z1: 4,
                },
                renderer_id: "test-vox",
                extra: Box::new(()),
            }])
        }
        fn manifest(&self) -> Vec<VolumeLabel> {
            // A label spanning the whole (full-res) box, in `vol_dim` voxel space
            // — the contract the viewer relies on (never rescaled to the coarse
            // grid at build time).
            let [ex, ey, ez] = self.extent;
            vec![VolumeLabel {
                name: "test-tensor".to_string(),
                group: "layer 0".to_string(),
                bbox: VoxelBox {
                    x0: 0,
                    y0: 0,
                    z0: 0,
                    x1: ex,
                    y1: ey,
                    z1: ez,
                },
            }]
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct TestVolPlugin;
    impl crate::registry::VolumeShapePlugin for TestVolPlugin {
        fn id(&self) -> &'static str {
            "test-vol"
        }
        fn priority(&self) -> i32 {
            1000
        }
        fn applicable(&self, _ctx: &crate::registry::LayoutBuildCtx<'_>) -> bool {
            true
        }
        fn build(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> Option<Box<dyn VolumeShape>> {
            // An anisotropic box derived from the requested resolution, to
            // exercise the non-cube path end to end.
            let s = ctx.grid_side;
            Some(Box::new(TestVolume {
                extent: [s, s * 2, s / 2],
            }))
        }
    }

    struct TestVox;
    impl VoxelRenderer for TestVox {
        fn id(&self) -> &'static str {
            "test-vox"
        }
        fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>) {
            // Prove arbvis fetched the entity span before dispatch.
            assert_eq!(ctx.bytes.len(), 16);
            let bb = ctx.entity.bbox;
            for z in bb.z0..bb.z1 {
                for y in bb.y0..bb.y1 {
                    for x in bb.x0..bb.x1 {
                        grid.put(
                            x,
                            y,
                            z,
                            VoxelCell {
                                r: 10,
                                g: 20,
                                b: 30,
                                a: 200,
                            },
                        );
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structured_entities_bake_rgb() {
        let mut reg = Registry::with_defaults();
        reg.volume_shapes.push(Arc::new(TestVolPlugin));
        reg.voxel.register_renderer(Arc::new(TestVox));

        let dir = tempfile::tempdir().unwrap();
        let grid_side = 8u32;
        render_volume(
            vec![buffered(vec![0u8; 16])],
            16,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            grid_side,
            0,
            LayoutMode::Auto,
            &reg,
            &Branding::default(),
        )
        .await
        .unwrap();

        // TestVolPlugin builds an anisotropic [s, 2s, s/2] box from grid_side.
        let expect_extent = [grid_side, grid_side * 2, grid_side / 2];
        let expect_cells =
            expect_extent[0] as u64 * expect_extent[1] as u64 * expect_extent[2] as u64;

        let vol = std::fs::read(dir.path().join("volume.bin")).unwrap();
        assert_eq!(vol.len() as u64, expect_cells * 4);
        let mut filled = 0u64;
        for px in vol.chunks_exact(4) {
            if px[3] == 200 {
                assert_eq!(
                    [px[0], px[1], px[2]],
                    [10, 20, 30],
                    "baked RGB survives verbatim"
                );
                filled += 1;
            }
        }
        assert_eq!(filled, 4 * 4 * 4, "the 4³ entity box should be baked");

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["color_mode"], "rgb");
        assert_eq!(
            meta["grid_extent"],
            serde_json::json!(expect_extent),
            "structured shape keeps its anisotropic box"
        );
    }

    /// A structured shape whose extent exceeds `COARSE_CAP` must now **stream**
    /// like the byte path: `color_mode == "rgb"`, `bricks.streamed == true`, a
    /// small aspect-preserving coarse `volume.bin`, the full anisotropic extent in
    /// `bricks.vol_dim`, an octree `tree.bin` (no flat page table), and a flat
    /// `bricks.bin` of `occupied · brick³ · 4` bytes. The manifest bbox stays in
    /// full `vol_dim` coords (never rescaled to the coarse grid).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structured_streams_above_coarse_cap() {
        let mut reg = Registry::with_defaults();
        reg.volume_shapes.push(Arc::new(TestVolPlugin));
        reg.voxel.register_renderer(Arc::new(TestVox));

        let dir = tempfile::tempdir().unwrap();
        // grid_side 130 → TestVolPlugin box [130, 260, 65]; max 260 > COARSE_CAP(256)
        // ⇒ streamed. Just over the cap keeps the coarse grid + slab buffer cheap.
        let grid_side = 130u32;
        render_volume(
            vec![buffered(vec![0u8; 16])],
            16,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            grid_side,
            0,
            LayoutMode::Auto,
            &reg,
            &Branding::default(),
        )
        .await
        .unwrap();

        let full_extent = [grid_side, grid_side * 2, grid_side / 2]; // [130,260,65]
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();

        assert_eq!(meta["color_mode"], "rgb", "structured stays RGB-baked");
        let bm = &meta["bricks"];
        assert_eq!(bm["streamed"], true, "above the cap ⇒ streamed pool");
        assert_eq!(bm["apron"], 0, "streamed bricks have no apron");
        assert_eq!(
            bm["vol_dim"],
            serde_json::json!(full_extent),
            "octree addresses the full anisotropic extent"
        );

        // Coarse volume.bin: aspect-preserving, longest axis == COARSE_CAP.
        let ce = &meta["grid_extent"];
        let ge = [
            ce[0].as_u64().unwrap() as u32,
            ce[1].as_u64().unwrap() as u32,
            ce[2].as_u64().unwrap() as u32,
        ];
        assert_eq!(
            ge,
            [128, 256, 64],
            "coarse grid preserves 2:4:1 aspect at cap 256"
        );
        assert!(ge.iter().max().unwrap() <= &COARSE_CAP);
        let vol = std::fs::read(dir.path().join("volume.bin")).unwrap();
        assert_eq!(
            vol.len() as u64,
            ge[0] as u64 * ge[1] as u64 * ge[2] as u64 * 4,
            "volume.bin is the small coarse grid, not the 15 GB full one"
        );

        // Octree page structure, no flat page table; flat brick blocks.
        assert!(
            dir.path().join("tree.bin").exists(),
            "streamed ships tree.bin"
        );
        assert!(
            !dir.path().join("pagetable.bin").exists(),
            "no flat page table"
        );
        let occupied = bm["occupied"].as_u64().unwrap();
        assert_eq!(occupied, 1, "the 4³ entity sits in a single brick");
        let brick = bm["brick"].as_u64().unwrap() as usize;
        let bricks = std::fs::read(dir.path().join("bricks.bin")).unwrap();
        assert_eq!(
            bricks.len() as u64,
            occupied * (brick.pow(3) * 4) as u64,
            "bricks.bin is a flat occupied-block array"
        );
        // The baked RGBA survives verbatim in the streamed block (no LUT channels).
        assert_eq!(
            &bricks[0..4],
            &[10, 20, 30, 200],
            "brick RGBA is baked color, verbatim"
        );

        // Manifest bbox stays in full vol_dim coords (client maps it, not the build).
        let mb = &meta["manifest"][0]["bbox"];
        assert_eq!(
            [
                mb["x1"].as_u64().unwrap(),
                mb["y1"].as_u64().unwrap(),
                mb["z1"].as_u64().unwrap()
            ],
            [
                full_extent[0] as u64,
                full_extent[1] as u64,
                full_extent[2] as u64
            ],
            "manifest bbox is NOT rescaled to the coarse grid"
        );
    }
}
