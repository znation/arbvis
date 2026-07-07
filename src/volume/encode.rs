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
    /// removed. `4` ⇒ volume-only (no point cloud).
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

/// Precompute the 0–255 luminance of each byte's LUT color, used as the
/// per-byte "activity" weight during aggregation.
pub fn luma_lut(pixel_lut: &[Rgb<u8>; 256]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for (i, c) in pixel_lut.iter().enumerate() {
        out[i] = (c.0[0] as u16 + c.0[1] as u16 + c.0[2] as u16) / 3;
    }
    out
}

