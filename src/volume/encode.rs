//! Encoding of the aggregated 3D volume into the on-disk bundle the Three.js
//! viewer consumes: a raw RGBA8 `Data3DTexture` payload and a JSON sidecar.

use image::Rgb;
use serde::Serialize;

/// Per-voxel accumulator filled during the byte→Hilbert aggregation pass.
/// One byte stream pass populates these; [`grid_to_rgba`] turns them into the
/// texture the shader samples.
#[derive(Clone, Copy, Default)]
pub struct VoxelAcc {
    /// Number of source bytes that mapped into this voxel.
    pub count: u32,
    /// Sum of raw byte values (→ mean value → color via the LUT in-shader).
    pub sum_val: u64,
    /// Sum of per-byte color luminance (→ mean luminance → the default
    /// "byte activity" opacity source; near-zero/padding reads as transparent).
    pub sum_luma: u64,
}

/// JSON sidecar describing the bundle. The viewer fetches this first, then
/// `volume.bin`, sizing its `Data3DTexture` from these fields. The byte→color
/// LUT travels here too so the shader colors voxels exactly like the 2D
/// renderer.
#[derive(Serialize)]
pub struct VolumeMeta {
    pub title: String,
    pub brand_name: String,
    pub repo_url: String,
    /// Grid box `[x, y, z]` in voxels; `volume.bin` holds `x*y*z` RGBA8 texels
    /// in x-fastest, then y, then z order. The byte path emits a cube (equal
    /// power-of-two sides); a structured path may emit an anisotropic box.
    pub grid_extent: [u32; 3],
    pub total_bytes: u64,
    /// Largest per-voxel byte count, used to normalize the "fill density"
    /// opacity channel back to a count on the client if desired.
    pub max_count: u64,
    pub diff_mode: bool,
    /// How the viewer colors voxels: `"lut"` (byte path — R indexes the
    /// shader LUT below) or `"rgb"` (structured path — RGB is baked final
    /// color, A is the opacity/occupancy weight). Absent in pre-seam bundles;
    /// the viewer defaults to `"lut"`.
    pub color_mode: String,
    pub inputs: Vec<String>,
    /// Center of the occupied region in viewer world space (the box is centered
    /// at the origin; its longest axis spans `[-0.5, 0.5]`), so the viewer can
    /// frame the data instead of the whole (often mostly-empty) box.
    pub focus_center: [f32; 3],
    /// Half-extent of the occupied region (world-space radius) for camera framing.
    pub focus_radius: f32,
    /// 256-entry byte→RGB lookup table (the active 2D LUT: plain or diff).
    pub lut: Vec<[u8; 3]>,
    /// Per-entity pick manifest (structured cubes only; empty for the byte
    /// floor). The viewer builds invisible pick boxes from these so a click can
    /// name the tensor under the cursor.
    pub manifest: Vec<super::shape::VolumeLabel>,
    /// Bundle format version. `3` ⇒ the brick volume may be ray-guided streamed
    /// (`bricks.streamed`); earlier versions also shipped a point cloud, since
    /// removed. `4` ⇒ volume-only (no point cloud). `5` ⇒ the streamed page
    /// structure is a sparse octree node pool (`bricks.tree_*`) instead of a flat
    /// page table.
    pub format_version: u32,
    /// Sparse brick pool + page table the volume ray-march renders from
    /// (GigaVoxels-style indirection, replacing the dense full-cube texture and
    /// the occupancy mip). Absent ⇒ the viewer renders the dense `volume.bin`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bricks: Option<BrickVolumeMeta>,
}

/// Descriptor for the sparse brick pool + page table (see [`super::brick`]).
/// The volume ray-march reads `page_file` to leap empty bricks and to find the
/// `atlas_file` texels for occupied ones.
#[derive(Serialize)]
pub struct BrickVolumeMeta {
    /// Brick-pool atlas, RGBA8, `atlas_dim` voxels.
    pub atlas_file: String,
    /// Page table, RGBA8, `page_dim` cells (1-based atlas slot in R,G,B).
    pub page_file: String,
    /// Brick edge in voxels.
    pub brick: u32,
    /// Page-table dims in bricks `[x, y, z]` (`ceil(vol_dim / brick)`).
    pub page_dim: [u32; 3],
    /// Atlas dims in voxels `[x, y, z]` (each a multiple of `brick`).
    pub atlas_dim: [u32; 3],
    /// Voxel extent the page table represents `[x, y, z]` (the virtual
    /// resolution; ≥ `grid_extent`). The viewer maps `uvw·vol_dim → voxel`.
    pub vol_dim: [u32; 3],
    /// Apron border per brick (stored brick edge = `brick + 2·apron`). `> 0` ⇒
    /// the viewer trilinearly filters across brick edges; `0` ⇒ nearest.
    pub apron: u32,
    /// Number of occupied bricks (`atlas_file` slots, or flat blocks when
    /// streamed).
    pub occupied: u32,
    /// `false` (or absent in pre-v3 bundles) ⇒ `atlas_file` is a packed 3D atlas
    /// (`atlas_dim` voxels) uploaded whole and the page table holds resident
    /// atlas slots. `true` ⇒ ray-guided streaming: `atlas_file` is a flat array
    /// of `occupied` bricks (`brick³·4` bytes each, brick id `S` at
    /// `(S-1)·brick³·4`), `atlas_dim` is unused, and `page_file` holds 1-based
    /// brick **ids**. The viewer streams bricks into a bounded GPU cache on
    /// demand, so VRAM is decoupled from the data's total size.
    #[serde(default)]
    pub streamed: bool,

    // --- Sparse octree page structure (streamed path; replaces the flat page
    // table so the page structure is O(occupied), not O((vol/brick)³)). Absent
    // (`tree_depth == 0`) in the non-streamed/flat path and in pre-v5 bundles.
    /// Octree node-pool file (RGBA8, `tree_dim` texels). Each node is a 2×2×2
    /// child-entry block; entry `A>0` ⇒ leaf (RGB = 1-based brick id, `A=1`),
    /// `A==0 && RGB>0` ⇒ internal (RGB = 1-based child-node index), all-zero ⇒
    /// empty subtree. Node 0 is the root over the whole `2^tree_depth` brick cube.
    #[serde(default)]
    pub tree_file: String,
    /// Node-pool texture dims in texels `[x, y, z]` (each even).
    #[serde(default)]
    pub tree_dim: [u32; 3],
    /// Octree depth `D = log2(bricks per side)`. `0` ⇒ no octree (flat page
    /// table); `> 0` ⇒ the viewer descends the node pool instead of `page_file`.
    #[serde(default)]
    pub tree_depth: u32,
    /// Octree node count (root + internal), for diagnostics.
    #[serde(default)]
    pub node_count: u32,
    /// Largest raw per-voxel count in the streamed byte atlas. `> 0` ⇒ the atlas
    /// B channel holds RAW counts and the viewer normalizes density by
    /// `255/max_count` when sampling a resident brick. `0`/absent ⇒ B is already
    /// baked (structured verbatim RGBA, non-streamed, or pre-v6 bundles) and the
    /// viewer does not rescale. Deferring this divide lets bricks stream to disk
    /// before the final max is known.
    #[serde(default)]
    pub max_count: u64,
}

/// Convert the accumulated grid into the RGBA8 texel buffer.
///
/// Channel layout per voxel (what the ray-march shader reads):
///   R = mean byte value      → color via the LUT
///   G = mean luminance        → "byte activity" opacity source (default)
///   B = normalized fill count → "fill density" opacity source (alternate)
///   A = occupancy mask        → 0 for empty voxels (fully skipped)
///
/// `grid` is indexed x-fastest (`x + y*side + z*side*side`) so it uploads
/// directly as a `THREE.Data3DTexture`.
pub fn grid_to_rgba(grid: &[VoxelAcc], max_count: u64) -> Vec<u8> {
    let inv_max = if max_count > 0 {
        255.0 / max_count as f64
    } else {
        0.0
    };
    let mut out = vec![0u8; grid.len() * 4];
    for (i, acc) in grid.iter().enumerate() {
        if acc.count == 0 {
            continue; // leave [0,0,0,0]: empty → transparent
        }
        let count = acc.count as u64;
        let mean_val = (acc.sum_val / count).min(255) as u8;
        let activity = (acc.sum_luma / count).min(255) as u8;
        let fill = ((acc.count as f64) * inv_max).round().clamp(0.0, 255.0) as u8;
        let o = i * 4;
        out[o] = mean_val;
        out[o + 1] = activity;
        out[o + 2] = fill;
        out[o + 3] = 255;
    }
    out
}

/// Pack a structured ([`super::voxel::VoxelCell`]) grid into the RGBA8 texel
/// buffer verbatim — the renderer already baked final color into R/G/B and the
/// opacity weight into A. Same x-fastest layout as [`grid_to_rgba`], so it
/// uploads directly as a `THREE.Data3DTexture`.
pub fn pack_voxel_cells(grid: &[super::voxel::VoxelCell]) -> Vec<u8> {
    let mut out = vec![0u8; grid.len() * 4];
    for (i, c) in grid.iter().enumerate() {
        let o = i * 4;
        out[o] = c.r;
        out[o + 1] = c.g;
        out[o + 2] = c.b;
        out[o + 3] = c.a;
    }
    out
}

/// Aspect-preserving coarse extent: scale so the longest axis lands at `cap`
/// (keeping the box's proportions), clamped to `[1, full]` per axis. Returns
/// `full` unchanged when it already fits within `cap`. Sizes the small dense
/// `volume.bin` (coarse fallback LOD + CPU pick/histograms) for the streamed
/// structured path while the fine detail streams from the brick pool.
pub fn coarse_extent(full: [u32; 3], cap: u32) -> [u32; 3] {
    let m = full[0].max(full[1]).max(full[2]);
    if m <= cap {
        return full;
    }
    let mut ce = [0u32; 3];
    for a in 0..3 {
        // Round to nearest, clamp to [1, full[a]]; the longest axis lands at cap.
        ce[a] = (((full[a] as u64 * cap as u64) + (m as u64 / 2)) / m as u64)
            .max(1)
            .min(full[a] as u64) as u32;
    }
    ce
}

/// Box-average downsample of an RGBA8 dense grid from `full_extent` to a smaller
/// `coarse` extent (both x-fastest). Color is the alpha-weighted mean of the
/// occupied source voxels in each coarse cell's block (premultiplied-correct, so
/// transparent voxels don't wash the color toward black); alpha is the
/// coverage-weighted mean over the whole block (sparse regions dim, like a real
/// LOD). Empty blocks stay `[0,0,0,0]`. This is a blurry fallback only visible
/// until the fine bricks stream in, so an approximate LOD is fine.
///
/// The production structured path folds slabs through [`CoarseAcc`] instead (to
/// stay dense-grid-free); this one-shot form is the reference both are tested
/// against.
#[cfg(test)]
pub fn downsample_rgba(full: &[u8], full_extent: [u32; 3], coarse: [u32; 3]) -> Vec<u8> {
    let [fx, fy, fz] = full_extent;
    let [cx, cy, cz] = coarse;
    let (fxs, fys) = (fx as usize, fy as usize);
    let mut out = vec![0u8; cx as usize * cy as usize * cz as usize * 4];
    for oz in 0..cz {
        let (z0, z1) = coarse_block(oz, cz, fz);
        for oy in 0..cy {
            let (y0, y1) = coarse_block(oy, cy, fy);
            for ox in 0..cx {
                let (x0, x1) = coarse_block(ox, cx, fx);
                let (mut sr, mut sg, mut sb) = (0f64, 0f64, 0f64);
                let mut wsum = 0f64; // Σ alpha over occupied voxels (color weight)
                let mut asum = 0f64; // Σ alpha over the whole block
                let mut n = 0u64; // block voxel count (coverage denominator)
                for z in z0..z1 {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let s = (x as usize + y as usize * fxs + z as usize * fxs * fys) * 4;
                            let a = full[s + 3] as f64;
                            if a > 0.0 {
                                sr += full[s] as f64 * a;
                                sg += full[s + 1] as f64 * a;
                                sb += full[s + 2] as f64 * a;
                                wsum += a;
                            }
                            asum += a;
                            n += 1;
                        }
                    }
                }
                if wsum > 0.0 {
                    let dst = (ox as usize
                        + oy as usize * cx as usize
                        + oz as usize * cx as usize * cy as usize)
                        * 4;
                    out[dst] = (sr / wsum).round().clamp(0.0, 255.0) as u8;
                    out[dst + 1] = (sg / wsum).round().clamp(0.0, 255.0) as u8;
                    out[dst + 2] = (sb / wsum).round().clamp(0.0, 255.0) as u8;
                    out[dst + 3] = (asum / n as f64).round().clamp(0.0, 255.0) as u8;
                }
                // else: fully transparent block → leave [0,0,0,0]
            }
        }
    }
    out
}

/// Streaming accumulator that reproduces [`downsample_rgba`] one **brick-aligned
/// Z-slab** at a time, so the structured path can build the coarse `volume.bin`
/// without ever holding the full dense grid. The alpha-weighted color sums and
/// coverage counters are additive over disjoint voxel subsets, so folding the
/// slabs that partition `[0, full.z)` yields a buffer byte-identical to a single
/// [`downsample_rgba`] over the whole grid.
pub struct CoarseAcc {
    full: [u32; 3],
    coarse: [u32; 3],
    sr: Vec<f64>,
    sg: Vec<f64>,
    sb: Vec<f64>,
    wsum: Vec<f64>, // Σ alpha over occupied voxels (color weight)
    asum: Vec<f64>, // Σ alpha over the whole block
    n: Vec<u64>,    // block voxel count (coverage denominator)
}

/// Source block `[lo, hi)` covering coarse index `i` on an axis of length f→c.
/// Shared by [`downsample_rgba`] and [`CoarseAcc`] so both partition identically.
fn coarse_block(i: u32, c: u32, f: u32) -> (u32, u32) {
    let lo = ((i as u64 * f as u64) / c as u64) as u32;
    let hi = ((((i as u64 + 1) * f as u64).div_ceil(c as u64)) as u32)
        .max(lo + 1)
        .min(f);
    (lo, hi)
}

impl CoarseAcc {
    pub fn new(full: [u32; 3], coarse: [u32; 3]) -> Self {
        let cells = coarse[0] as usize * coarse[1] as usize * coarse[2] as usize;
        CoarseAcc {
            full,
            coarse,
            sr: vec![0.0; cells],
            sg: vec![0.0; cells],
            sb: vec![0.0; cells],
            wsum: vec![0.0; cells],
            asum: vec![0.0; cells],
            n: vec![0; cells],
        }
    }

    /// Fold one brick-aligned Z-slab (RGBA8, `full.x·full.y·(z1-z0)`, x-fastest,
    /// planes z-relative to `z0`) into the accumulator. Each coarse-z block is
    /// clipped to `[z0, z1)`; across the slabs that tile `[0, full.z)` every
    /// source voxel is counted exactly once.
    pub fn add_slab(&mut self, slab: &[u8], z0: u32, z1: u32) {
        let [fx, fy, fz] = self.full;
        let [cx, cy, cz] = self.coarse;
        let (fxs, fys) = (fx as usize, fy as usize);
        for oz in 0..cz {
            let (zb0, zb1) = coarse_block(oz, cz, fz);
            let (zlo, zhi) = (zb0.max(z0), zb1.min(z1)); // block ∩ slab
            if zlo >= zhi {
                continue;
            }
            for oy in 0..cy {
                let (y0, y1) = coarse_block(oy, cy, fy);
                for ox in 0..cx {
                    let (x0, x1) = coarse_block(ox, cx, fx);
                    let ci = ox as usize
                        + oy as usize * cx as usize
                        + oz as usize * cx as usize * cy as usize;
                    for z in zlo..zhi {
                        let zl = (z - z0) as usize;
                        for y in y0..y1 {
                            for x in x0..x1 {
                                let s = (x as usize + y as usize * fxs + zl * fxs * fys) * 4;
                                let a = slab[s + 3] as f64;
                                if a > 0.0 {
                                    self.sr[ci] += slab[s] as f64 * a;
                                    self.sg[ci] += slab[s + 1] as f64 * a;
                                    self.sb[ci] += slab[s + 2] as f64 * a;
                                    self.wsum[ci] += a;
                                }
                                self.asum[ci] += a;
                                self.n[ci] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Finalize to the coarse RGBA8 buffer — same per-cell math as
    /// [`downsample_rgba`] (alpha-weighted color, coverage-weighted alpha).
    pub fn finish(self) -> Vec<u8> {
        let [cx, cy, cz] = self.coarse;
        let mut out = vec![0u8; cx as usize * cy as usize * cz as usize * 4];
        for ci in 0..out.len() / 4 {
            let w = self.wsum[ci];
            if w > 0.0 {
                let dst = ci * 4;
                out[dst] = (self.sr[ci] / w).round().clamp(0.0, 255.0) as u8;
                out[dst + 1] = (self.sg[ci] / w).round().clamp(0.0, 255.0) as u8;
                out[dst + 2] = (self.sb[ci] / w).round().clamp(0.0, 255.0) as u8;
                out[dst + 3] =
                    (self.asum[ci] / self.n[ci] as f64).round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

/// Precompute the 0–255 luminance of each byte's LUT color, used as the
/// per-byte "activity" weight during aggregation.
pub fn luma_lut(pixel_lut: &[Rgb<u8>; 256]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for (i, c) in pixel_lut.iter().enumerate() {
        out[i] = (c.0[0] as u16 + c.0[1] as u16 + c.0[2] as u16) / 3;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_extent_preserves_aspect_and_caps_longest_axis() {
        // Already within cap → unchanged.
        assert_eq!(coarse_extent([64, 32, 16], 128), [64, 32, 16]);
        // Longest axis → cap, proportions kept (2:4:1).
        assert_eq!(coarse_extent([80, 160, 40], 128), [64, 128, 32]);
        // Cube.
        assert_eq!(coarse_extent([256, 256, 256], 128), [128, 128, 128]);
        // Never zero on a thin axis.
        assert_eq!(coarse_extent([2048, 4, 4], 128)[1], 1);
    }

    #[test]
    fn downsample_rgba_alpha_weights_color_and_covers_alpha() {
        // 2 voxels → 1 coarse cell: color is the alpha-weighted mean of occupied
        // voxels; alpha is the coverage mean over the whole block.
        // (240·200 + 40·100)/(200+100) = 173.33 → 173; alpha (200+100)/2 = 150.
        let full = [2u32, 1, 1];
        let g = [240u8, 0, 0, 200, 40, 0, 0, 100];
        assert_eq!(downsample_rgba(&g, full, [1, 1, 1]), vec![173, 0, 0, 150]);

        // A transparent voxel must not drag the color toward black (premultiplied
        // weighting): color stays the occupied voxel's, alpha halves for coverage.
        let g2 = [100u8, 0, 0, 200, 0, 0, 0, 0];
        assert_eq!(downsample_rgba(&g2, full, [1, 1, 1]), vec![100, 0, 0, 100]);

        // Fully-transparent block → [0,0,0,0].
        let g3 = [0u8; 8];
        assert_eq!(downsample_rgba(&g3, full, [1, 1, 1]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn coarse_acc_slabs_match_downsample() {
        // Fold an anisotropic grid in Z-slabs and confirm the result is
        // byte-identical to downsample_rgba over the whole grid — including a
        // coarse-z block ([7,10)) that straddles the slab boundary at z=8.
        let full = [4u32, 3, 10];
        let coarse = [2u32, 2, 3]; // z blocks: [0,4), [4,7), [7,10)
        let (fx, fy, fz) = (4usize, 3, 10);
        let mut g = vec![0u8; fx * fy * fz * 4];
        let mut s: u64 = 0x1234_5678;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for px in g.chunks_exact_mut(4) {
            px[0] = (next() % 256) as u8;
            px[1] = (next() % 256) as u8;
            px[2] = (next() % 256) as u8;
            px[3] = if next() % 3 == 0 { 0 } else { (next() % 255 + 1) as u8 };
        }
        let reference = downsample_rgba(&g, full, coarse);

        let mut acc = CoarseAcc::new(full, coarse);
        for (z0, z1) in [(0u32, 8u32), (8u32, 10u32)] {
            let depth = (z1 - z0) as usize;
            let mut slab = vec![0u8; fx * fy * depth * 4];
            for z in z0..z1 {
                let (src, dst) = ((z as usize) * fx * fy * 4, (z - z0) as usize * fx * fy * 4);
                slab[dst..dst + fx * fy * 4].copy_from_slice(&g[src..src + fx * fy * 4]);
            }
            acc.add_slab(&slab, z0, z1);
        }
        assert_eq!(acc.finish(), reference, "slab accumulation == full downsample");
    }

    #[test]
    fn downsample_rgba_splits_blocks_evenly() {
        // 4→2 on x: coarse[0] covers src x∈[0,2), coarse[1] covers x∈[2,4).
        let full = [4u32, 1, 1];
        let mut g = vec![0u8; 4 * 4];
        g[0..4].copy_from_slice(&[100, 0, 0, 200]); // x0 (in coarse 0)
        g[12..16].copy_from_slice(&[50, 0, 0, 100]); // x3 (in coarse 1)
        let out = downsample_rgba(&g, full, [2, 1, 1]);
        assert_eq!(out, vec![100, 0, 0, 100, /*|*/ 50, 0, 0, 50]);
    }
}

