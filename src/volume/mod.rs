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

pub mod brick;
pub mod encode;
pub mod html;
pub mod shape;
pub mod voxel;

pub use shape::{
    select_volume_shape, HilbertVolumePlugin, VolumeEntity, VolumeLabel, VolumeShape, VoxelBox,
};
pub use voxel::{VoxelCell, VoxelGridMut, VoxelRegistry, VoxelRenderCtx, VoxelRenderer};

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use image::Rgb;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Source};
use crate::geometry;
use crate::layout::LayoutMode;
use crate::registry::{Branding, Registry};
use encode::{VolumeMeta, VoxelAcc};

/// Bytes read per `fetch_range` window during aggregation.
const CHUNK: u64 = 4 * 1024 * 1024;

/// The dense `volume.bin` (coarse fallback LOD + CPU pick/histogram buffer) is
/// capped at this side so the mandatory up-front download stays small and fixed
/// (128³·4 ≈ 8 MiB) regardless of the requested detail resolution. Detail finer
/// than this streams on demand from the sparse brick pool.
pub(crate) const COARSE_CAP: u32 = 128;

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

struct BuildResult {
    volume_rgba: Vec<u8>,
    /// Extent of the grid actually written to `volume.bin` (`volume_rgba`). Equals
    /// the bake extent for the dense paths, but the **coarse** extent when the
    /// structured path streams (the full detail lives in `bricks`, and `volume.bin`
    /// is a small aspect-preserving downsample for the fallback LOD + CPU pick).
    grid_extent: [u32; 3],
    max_count: u64,
    focus_center: [f32; 3],
    focus_radius: f32,
    /// Sparse brick pool built at a higher virtual resolution (byte floor with
    /// `--volume-res`, or the streamed structured path); `None` ⇒ `render_volume`
    /// derives bricks from the dense grid instead.
    bricks: Option<brick::BrickVolume>,
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
    let actual_extent = if is_byte { [dense_side; 3] } else { shape.grid_extent() };
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
            if structured_streamed { " (streamed)" } else { "" }
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
            aggregate_bytes_hilbert(sources, total, ex, volume_res, pixel_lut, rt, &out_dir_build)
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

    std::fs::write(out_dir.join("volume.bin"), &built.volume_rgba)?;

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
        std::fs::write(out_dir.join("bricks.bin"), &bricks.atlas)?;
    }
    // Page structure: the streamed path ships a sparse octree node pool
    // (`tree.bin`); the non-streamed/flat path ships the dense page table
    // (`pagetable.bin`).
    let (page_file, tree_file) = if bricks.streamed {
        std::fs::write(out_dir.join("tree.bin"), &bricks.node_pool)?;
        (String::new(), "tree.bin".to_string())
    } else {
        std::fs::write(out_dir.join("pagetable.bin"), &bricks.page_table)?;
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
    std::fs::write(out_dir.join("meta.json"), serde_json::to_vec(&meta)?)?;
    std::fs::write(
        out_dir.join("index.html"),
        html::build_volume_html(title, inputs, branding),
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
/// without re-aggregating. Mirrors [`crate::tiled::regen_html`].
pub fn regen_html(dir: &Path, branding: &Branding) -> anyhow::Result<()> {
    let meta_path = dir.join("meta.json");
    let bytes = std::fs::read(&meta_path)
        .with_context(|| format!("reading {} (is this a --3d bundle?)", meta_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let title = v
        .get("title")
        .and_then(|t| t.as_str())
        .map(String::from)
        .unwrap_or_else(|| branding.name.to_string());
    let inputs: Vec<String> = v
        .get("inputs")
        .and_then(|i| i.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    std::fs::write(
        dir.join("index.html"),
        html::build_volume_html(&title, &inputs, branding),
    )?;
    log::info!("Regenerated 3D viewer index.html in {}", dir.display());
    Ok(())
}

/// Byte-Hilbert floor: a single streaming pass over the concatenated source
/// bytes that populates the voxel grid (and, with `--volume-res`, the
/// higher-resolution sparse brick pool). Runs on the blocking pool.
fn aggregate_bytes_hilbert(
    sources: Vec<Source>,
    total: u64,
    grid_side: u32,
    volume_res: u32,
    pixel_lut: [Rgb<u8>; 256],
    rt: tokio::runtime::Handle,
    out_dir: &Path,
) -> anyhow::Result<BuildResult> {
    let order = grid_side.trailing_zeros(); // grid_side is a power of two
    let cells: u64 = (grid_side as u64).pow(3);
    let luma = encode::luma_lut(&pixel_lut);

    let mut grid = vec![VoxelAcc::default(); cells as usize];
    let mut max_count: u64 = 0;

    // Optional higher-resolution sparse brick pool (--volume-res > --grid).
    // Built streaming in Hilbert order — one open brick at a time, O(brick)
    // memory — so the *volume* can exceed the dense grid for sparse data. The
    // dense grid above is still built (coarser) for histograms + pick.
    let order_v = if volume_res > grid_side {
        volume_res.trailing_zeros()
    } else {
        0
    };
    let cells_v: u128 = if order_v > 0 { 1u128 << (3 * order_v) } else { 0 };
    // The streamed brick pool writes each finished brick straight to bricks.bin
    // (append-only, O(one brick) RAM) as the Hilbert curve advances.
    let mut brick_builder = if order_v > 0 {
        let w = std::io::BufWriter::new(std::fs::File::create(out_dir.join("bricks.bin"))?);
        Some(brick::BrickBuilder::new(order_v, brick::BRICK, luma, w))
    } else {
        None
    };

    // First byte not belonging to cell `c`. Two regimes: when the cube can't
    // hold every byte (`total > cells`) each cell aggregates a contiguous byte
    // run; otherwise each byte is its own cell (a contiguous Hilbert prefix, so
    // a small file reads as one solid blob rather than a scattered dust).
    let cell_end = |c: u64| -> u64 {
        if total <= cells {
            c + 1
        } else {
            (((c as u128 + 1) * total as u128).div_ceil(cells as u128)) as u64
        }
    };

    let side = grid_side as usize;
    let flush = |grid: &mut [VoxelAcc], c: u64, acc: &VoxelAcc, max_count: &mut u64| {
        if acc.count == 0 {
            return;
        }
        let [x, y, z] = geometry::hilbert_d2xyz(c, order);
        let lin = x as usize + y as usize * side + z as usize * side * side;
        grid[lin] = *acc;
        if acc.count as u64 > *max_count {
            *max_count = acc.count as u64;
        }
    };

    let mut cur_cell: u64 = 0;
    let mut ce = if total == 0 { 0 } else { cell_end(0) };
    let mut acc = VoxelAcc::default();
    let mut global_start: u64 = 0;

    for src in &sources {
        let data = load_source_data(src)?;
        let size = src.byte_size;
        let mut local: u64 = 0;
        while local < size {
            let len = (size - local).min(CHUNK) as usize;
            let buf = rt.block_on(data.fetch_range(local, len))?;
            for (i, &b) in buf.iter().enumerate() {
                let g = global_start + local + i as u64;

                // Advance the grid cell, flushing the one we leave behind.
                while g >= ce {
                    flush(&mut grid, cur_cell, &acc, &mut max_count);
                    acc = VoxelAcc::default();
                    cur_cell += 1;
                    ce = cell_end(cur_cell);
                }
                acc.count += 1;
                acc.sum_val += b as u64;
                acc.sum_luma += luma[b as usize] as u64;

                // Feed the high-resolution sparse brick pool (every byte, mapped
                // to its voxel on the 2^order_v cube).
                if let Some(bb) = brick_builder.as_mut() {
                    let cp_v = if total as u128 <= cells_v {
                        g
                    } else {
                        ((g as u128) * cells_v / total as u128) as u64
                    };
                    bb.push(cp_v, b);
                }
            }
            local += len as u64;
        }
        global_start += size;
    }
    // Flush the trailing in-progress cell.
    flush(&mut grid, cur_cell, &acc, &mut max_count);

    let volume_rgba = encode::grid_to_rgba(&grid, max_count);
    // Finish the streamed pool: flush bricks.bin and assemble the octree. Bricks
    // were already written to disk, so nothing is dropped — a full disk surfaces
    // as an IO error here rather than truncating detail.
    let bricks = match brick_builder {
        Some(bb) => Some(bb.finish_streaming()?.0),
        None => None,
    };

    // Camera framing. Stream the *fine* focus (the octree builder's occupied
    // bbox): for a small file at high resolution the coarse grid fills the whole
    // cube while the fine data is a tiny Hilbert-prefix corner, so framing from
    // the coarse grid would aim the camera at empty space. Non-streamed frames
    // from the dense grid as before.
    let (focus_center, focus_radius) = match &bricks {
        Some(bv) if bv.streamed => (bv.focus_center, bv.focus_radius),
        _ => occupied_focus(&grid, [grid_side; 3]),
    };

    Ok(BuildResult {
        volume_rgba,
        // The dense `volume.bin` is the (capped) coarse cube; the streamed pool
        // carries the fine detail at its own `vol_dim`.
        grid_extent: [grid_side; 3],
        max_count,
        focus_center,
        focus_radius,
        bricks,
    })
}

/// Structured path: render each entity into its voxel box via the matching
/// [`VoxelRenderer`], which bakes final RGBA8 straight into the grid (no shader
/// LUT). arbvis owns the fetch — it reads the whole `[byte_start, +byte_len)`
/// span per entity and hands the bytes to the renderer, which decodes/samples
/// within. (Peak memory is the largest single entity; a future revision can
/// switch to a fetch-on-demand callback for very large tensors.)
#[allow(clippy::too_many_arguments)]
fn aggregate_entities(
    sources: Vec<Source>,
    shape: &dyn VolumeShape,
    extent: [u32; 3],
    voxel_reg: &VoxelRegistry,
    diff_mode: bool,
    stream: bool,
    rt: tokio::runtime::Handle,
    out_dir: &Path,
) -> anyhow::Result<BuildResult> {
    let entities = shape.entities().unwrap_or_default();
    let default_renderer_id = shape.id();

    let resolve = |ent: &VolumeEntity| -> anyhow::Result<std::sync::Arc<dyn VoxelRenderer>> {
        voxel_reg
            .renderer(ent.renderer_id)
            .or_else(|| voxel_reg.renderer(default_renderer_id))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no voxel renderer registered for id `{}` (shape `{}`)",
                    ent.renderer_id,
                    default_renderer_id
                )
            })
    };

    if stream {
        // Dense-grid-free streamed path: process the volume in brick-aligned
        // Z-slabs so peak RAM is one slab (never `extent³`). Per slab, render
        // only the entities whose bbox intersects it into a slab-sized buffer,
        // brick that slab straight to `bricks.bin`, and fold it into the coarse
        // downsample. Frame from the fine occupied region (the aggregator's bbox).
        let [ex, ey, ez] = extent;
        let brick = brick::BRICK;

        // slab_depth: bound the slab buffer to ~SLAB_BUDGET_BYTES, a multiple of
        // BRICK (so no brick row straddles a boundary), clamped to [BRICK, ez↑].
        let layer = ex as usize * ey as usize * std::mem::size_of::<VoxelCell>();
        let raw = (SLAB_BUDGET_BYTES / layer.max(1)) as u32;
        let max_depth = ez.div_ceil(brick) * brick; // ez rounded up to a brick multiple
        let slab_depth = (raw / brick * brick).clamp(brick, max_depth);

        // Each entity's z-interval from its bbox (its authoritative target region,
        // known before dispatch), for scheduling and residency eviction.
        let ivals: Vec<(u32, u32)> = entities.iter().map(|e| (e.bbox.z0, e.bbox.z1)).collect();

        // Fetch-once residency: fetch an entity's bytes on the first slab it
        // intersects and evict once the slab front passes its bbox.z1. Live bytes
        // = Σ spans of entities overlapping the current slab.
        let mut cache: std::collections::HashMap<usize, std::sync::Arc<Vec<u8>>> =
            std::collections::HashMap::new();

        let mut agg = brick::StreamBrickAgg::new(extent, brick);
        let mut w = std::io::BufWriter::new(std::fs::File::create(out_dir.join("bricks.bin"))?);
        let ce = encode::coarse_extent(extent, COARSE_CAP);
        let mut coarse = encode::CoarseAcc::new(extent, ce);

        let mut z0 = 0u32;
        while z0 < ez {
            let z1 = (z0 + slab_depth).min(ez);
            let depth = (z1 - z0) as usize;
            let mut slab = vec![VoxelCell::default(); ex as usize * ey as usize * depth];

            for (i, ent) in entities.iter().enumerate() {
                let (ez0, ez1) = ivals[i];
                if ez0 >= z1 || ez1 <= z0 {
                    continue; // bbox doesn't intersect this slab
                }
                let renderer = resolve(ent)?;
                let bytes = match cache.get(&i) {
                    Some(b) => b.clone(),
                    None => {
                        let b = std::sync::Arc::new(fetch_entity_bytes(&sources, ent, &rt)?);
                        cache.insert(i, b.clone());
                        b
                    }
                };
                let ctx = VoxelRenderCtx {
                    entity: ent,
                    bytes: &bytes[..],
                    extent,
                    diff_mode,
                };
                let mut view = VoxelGridMut::slab(&mut slab, extent, z0, z1);
                renderer.render_window(&ctx, &mut view, z0..z1);
            }

            let slab_rgba = encode::pack_voxel_cells(&slab);
            agg.add_slab(&slab_rgba, z0, z1, &mut w)?;
            coarse.add_slab(&slab_rgba, z0, z1);

            cache.retain(|&i, _| ivals[i].1 > z1); // evict entities behind the front
            z0 = z1; // free `slab`/`slab_rgba`
        }
        w.flush()?;
        let bricks = agg.finish();
        return Ok(BuildResult {
            volume_rgba: coarse.finish(),
            grid_extent: ce,
            max_count: 0,
            focus_center: bricks.focus_center,
            focus_radius: bricks.focus_radius,
            bricks: Some(bricks),
        });
    }

    // Below the coarse cap: simple dense (non-streamed) path — render every
    // entity into one full grid (bounded by `extent ≤ COARSE_CAP`), then derive
    // bricks from it in render_volume.
    let cells = extent[0] as usize * extent[1] as usize * extent[2] as usize;
    let mut grid = vec![VoxelCell::default(); cells];
    for ent in &entities {
        let renderer = resolve(ent)?;
        let bytes = fetch_entity_bytes(&sources, ent, &rt)?;
        let ctx = VoxelRenderCtx {
            entity: ent,
            bytes: &bytes,
            extent,
            diff_mode,
        };
        let mut view = VoxelGridMut::new(&mut grid, extent);
        renderer.render(&ctx, &mut view);
    }

    let full_rgba = encode::pack_voxel_cells(&grid);
    let (focus_center, focus_radius) = shape
        .focus()
        .unwrap_or_else(|| occupied_focus_cells(&grid, extent));

    Ok(BuildResult {
        volume_rgba: full_rgba,
        grid_extent: extent,
        max_count: 0,
        focus_center,
        focus_radius,
        bricks: None,
    })
}

/// Fetch an entity's `[byte_start, +byte_len)` span (chunked, on the blocking
/// pool via `rt.block_on`). Peak transient RAM is the entity's own span.
fn fetch_entity_bytes(
    sources: &[Source],
    ent: &VolumeEntity,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Vec<u8>> {
    let src = sources.get(ent.source_idx).ok_or_else(|| {
        anyhow::anyhow!(
            "entity source_idx {} out of range ({} sources)",
            ent.source_idx,
            sources.len()
        )
    })?;
    let data = load_source_data(src)?;
    let mut bytes = vec![0u8; ent.byte_len as usize];
    let mut off = 0u64;
    while off < ent.byte_len {
        let len = (ent.byte_len - off).min(CHUNK) as usize;
        let chunk = rt.block_on(data.fetch_range(ent.byte_start + off, len))?;
        bytes[off as usize..off as usize + len].copy_from_slice(&chunk);
        off += len as u64;
    }
    Ok(bytes)
}

/// World-space framing center + radius from an occupied bounding box (per-axis
/// `bmin`/`bmax`, centroid `sum/n`) inside a grid of `extent` voxels. Targets
/// the occupancy **centroid** (mass center — the Hilbert prefix clusters
/// asymmetrically within its bounding box) and sizes the radius so the full
/// occupied bounding box stays in view from that center.
///
/// Voxel `v` on axis `a` maps to world position `((v + 0.5)/extent[a] - 0.5) *
/// scale[a]`, where `scale[a] = extent[a]/max(extent)` is the box's world size
/// on that axis (the viewer scales its longest axis to the unit cube and keeps
/// voxels cubic — matching the shader's `uvw = p/uSize + 0.5`). For a cube all
/// scales are 1, so this reduces to the old `(v + 0.5)/side - 0.5`.
pub(super) fn box_focus(
    bmin: [u32; 3],
    bmax: [u32; 3],
    sum: [f64; 3],
    n: u64,
    extent: [u32; 3],
) -> ([f32; 3], f32) {
    if n == 0 {
        return ([0.0, 0.0, 0.0], 0.5);
    }
    let maxext = extent[0].max(extent[1]).max(extent[2]) as f32;
    let to_world = |v: f32, axis: usize| {
        let e = extent[axis] as f32;
        ((v + 0.5) / e - 0.5) * (e / maxext)
    };
    let mut center = [0f32; 3];
    let mut radius = 0f32;
    for a in 0..3 {
        let c = to_world((sum[a] / n as f64) as f32, a);
        let lo = to_world(bmin[a] as f32, a);
        let hi = to_world(bmax[a] as f32, a);
        center[a] = c;
        radius = radius.max((c - lo).max(hi - c));
    }
    (center, radius.max(0.02))
}

/// Decode the x-fastest linear index `i` into voxel coordinates for `extent`.
fn voxel_coord(i: usize, extent: [u32; 3]) -> [u32; 3] {
    let (ex, ey) = (extent[0] as usize, extent[1] as usize);
    [
        (i % ex) as u32,
        ((i / ex) % ey) as u32,
        (i / (ex * ey)) as u32,
    ]
}

/// World-space framing for a structured (baked-RGBA) grid — occupancy is `a > 0`.
/// Mirrors [`occupied_focus`] but reads [`VoxelCell`] instead of [`VoxelAcc`].
fn occupied_focus_cells(grid: &[VoxelCell], extent: [u32; 3]) -> ([f32; 3], f32) {
    let mut bmin = [u32::MAX; 3];
    let mut bmax = [0u32; 3];
    let mut sum = [0f64; 3];
    let mut n: u64 = 0;
    for (i, cell) in grid.iter().enumerate() {
        if cell.a == 0 {
            continue;
        }
        n += 1;
        let coord = voxel_coord(i, extent);
        for a in 0..3 {
            bmin[a] = bmin[a].min(coord[a]);
            bmax[a] = bmax[a].max(coord[a]);
            sum[a] += coord[a] as f64;
        }
    }
    box_focus(bmin, bmax, sum, n, extent)
}

/// World-space framing center + radius for the occupied voxels of the byte grid.
/// Falls back to the whole box when nothing is occupied. See [`box_focus`] for
/// the voxel→world mapping.
fn occupied_focus(grid: &[VoxelAcc], extent: [u32; 3]) -> ([f32; 3], f32) {
    let mut bmin = [u32::MAX; 3];
    let mut bmax = [0u32; 3];
    let mut sum = [0f64; 3];
    let mut n: u64 = 0;
    for (i, acc) in grid.iter().enumerate() {
        if acc.count == 0 {
            continue;
        }
        n += 1;
        let coord = voxel_coord(i, extent);
        for a in 0..3 {
            bmin[a] = bmin[a].min(coord[a]);
            bmax[a] = bmax[a].max(coord[a]);
            sum[a] += coord[a] as f64;
        }
    }
    box_focus(bmin, bmax, sum, n, extent)
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
        assert_eq!(bm["vol_dim"][0], 16, "octree sized to the virtual resolution");
        assert_eq!(bm["tree_depth"], 1, "depth = log2(16 / brick=8) = 1");
        let occupied = bm["occupied"].as_u64().unwrap();
        assert!(occupied > 0, "some bricks are occupied");

        // bricks.bin is a flat array of occupied blocks; the sparse octree node
        // pool (tree.bin) indexes them — no flat pagetable.bin in the streamed path.
        let bricks = std::fs::read(dir.path().join("bricks.bin")).unwrap();
        let brick = bm["brick"].as_u64().unwrap() as usize;
        assert_eq!(bricks.len() as u64, occupied * (brick.pow(3) * 4) as u64,
            "bricks.bin holds occupied flat brick blocks");
        assert!(!dir.path().join("pagetable.bin").exists(), "streamed path emits no flat page table");
        let tree = std::fs::read(dir.path().join("tree.bin")).unwrap();
        let td = &bm["tree_dim"];
        let texels = td[0].as_u64().unwrap() * td[1].as_u64().unwrap() * td[2].as_u64().unwrap();
        assert_eq!(tree.len() as u64, texels * 4, "node pool is tree_dim texels of RGBA8");
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
                bbox: VoxelBox { x0: 0, y0: 0, z0: 0, x1: ex, y1: ey, z1: ez },
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
        // grid_side 80 → TestVolPlugin box [80, 160, 40]; max 160 > COARSE_CAP(128)
        // ⇒ streamed. ~512k cells (2 MiB) keeps the test cheap.
        let grid_side = 80u32;
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

        let full_extent = [grid_side, grid_side * 2, grid_side / 2]; // [80,160,40]
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
        assert_eq!(ge, [64, 128, 32], "coarse grid preserves 2:4:1 aspect at cap 128");
        assert!(ge.iter().max().unwrap() <= &COARSE_CAP);
        let vol = std::fs::read(dir.path().join("volume.bin")).unwrap();
        assert_eq!(
            vol.len() as u64,
            ge[0] as u64 * ge[1] as u64 * ge[2] as u64 * 4,
            "volume.bin is the small coarse grid, not the 15 GB full one"
        );

        // Octree page structure, no flat page table; flat brick blocks.
        assert!(dir.path().join("tree.bin").exists(), "streamed ships tree.bin");
        assert!(!dir.path().join("pagetable.bin").exists(), "no flat page table");
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
        assert_eq!(&bricks[0..4], &[10, 20, 30, 200], "brick RGBA is baked color, verbatim");

        // Manifest bbox stays in full vol_dim coords (client maps it, not the build).
        let mb = &meta["manifest"][0]["bbox"];
        assert_eq!(
            [mb["x1"].as_u64().unwrap(), mb["y1"].as_u64().unwrap(), mb["z1"].as_u64().unwrap()],
            [full_extent[0] as u64, full_extent[1] as u64, full_extent[2] as u64],
            "manifest bbox is NOT rescaled to the coarse grid"
        );
    }
}
