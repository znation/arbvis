//! Encoding of the aggregated 3D volume into the on-disk bundle the Three.js
//! viewer consumes: a raw RGBA8 `Data3DTexture` payload, an interleaved point
//! buffer, and a JSON sidecar.

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
/// `volume.bin` / `points.bin`, sizing its `Data3DTexture` and point buffer
/// from these fields. The byte→color LUT travels here too so the shader colors
/// voxels exactly like the 2D renderer.
#[derive(Serialize)]
pub struct VolumeMeta {
    pub title: String,
    pub brand_name: String,
    pub repo_url: String,
    /// Grid box `[x, y, z]` in voxels; `volume.bin` holds `x*y*z` RGBA8 texels
    /// in x-fastest, then y, then z order. The byte path emits a cube (equal
    /// power-of-two sides); a structured path may emit an anisotropic box.
    pub grid_extent: [u32; 3],
    /// Number of points in `points.bin`.
    pub points: u64,
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
    /// Bundle format version. Absent/`1` ⇒ wholesale only (the original
    /// bundle). `2` ⇒ also ships a streamed point-LOD octree (`point_octree`).
    pub format_version: u32,
    /// Streamed point-LOD octree descriptor; absent on wholesale-only bundles
    /// (older bundles, structured layouts, tiny inputs) so the viewer falls
    /// back to the wholesale `points.bin`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_octree: Option<PointOctreeMeta>,
}

/// Descriptor for the streamed point-LOD octree (the 3D analog of the 2D tile
/// pyramid). The viewer fetches `hierarchy_file` once, rebuilds the implicit
/// octree, then range-fetches node blocks from `data_file` on demand as the
/// camera refines — converging to one point per byte when zoomed in.
#[derive(Serialize)]
pub struct PointOctreeMeta {
    /// Concatenated per-node point blocks (each: 3 local coords + RGBA8).
    pub data_file: String,
    /// Fixed-size [`super::octree::NodeRecord`]s, `record_size` bytes each.
    pub hierarchy_file: String,
    pub record_size: u32,
    pub node_count: u64,
    /// Hilbert order `P`: the virtual point grid is `2^P` per axis.
    pub order: u32,
    /// Occupancy-grid exponent used per node (node cap = `(2^grid_log2)³`).
    pub grid_log2: u32,
    /// Total stored (post-LOD) points across all nodes.
    pub total_points: u64,
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

/// Pack the point cloud into one buffer: `positions` (f32 ×3 per point, in
/// `[0,1]`) immediately followed by `colors` (u8 ×4 per point, RGBA). The
/// viewer slices the two blocks using the point count from [`VolumeMeta`].
pub fn pack_points(positions: &[f32], colors: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(positions.len() * 4 + colors.len());
    for &f in positions {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out.extend_from_slice(colors);
    out
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
