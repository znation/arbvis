//! 3D coloring seam: the voxel analog of [`crate::tiled::leaf_renderer`].
//!
//! The byte-Hilbert floor ([`super::shape::HilbertVolume`]) colors voxels by
//! aggregating raw byte values and looking the mean up through a 256-entry LUT
//! *in the viewer shader* (`color_mode: "lut"`). A structured volume layout
//! (e.g. modelweightvis's `"arch"`) instead decodes each entity's elements by
//! dtype and bakes the final RGB straight into the voxel (`color_mode: "rgb"`),
//! because per-element decode needs more than one LUT (literal-byte vs
//! magnitude vs signed-diff) — which a single shader LUT can't express. A
//! [`VoxelRenderer`], registered by id like a [`crate::LeafRenderer`], owns that
//! decode-aggregate-colormap step; arbvis owns the fetch and the grid.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use super::shape::VolumeEntity;

/// Final per-voxel RGBA8 a structured [`VoxelRenderer`] writes into the cube.
///
/// In `"rgb"` color mode (structured layouts) `r`/`g`/`b` are the baked color
/// and `a` is the opacity/occupancy weight (the viewer uses `a` as both the
/// ray-march opacity source and the empty-voxel mask). An all-zero cell is an
/// empty voxel and is never rendered.
#[derive(Clone, Copy, Default)]
pub struct VoxelCell {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// Bounds-checked mutable view over the box a [`VoxelRenderer`] writes into.
///
/// Indexed x-fastest (`x + y*ex + z*ex*ey`, with `extent = [ex, ey, ez]`),
/// matching the byte path's grid so both pack into the same
/// `THREE.Data3DTexture` layout. Out-of-range `put`s are silently dropped — a
/// renderer can clamp loosely without panicking the run.
///
/// The view may cover only a **Z-slab** `[z0, z1)` of the full box (the
/// streamed structured path bricks the volume one slab at a time): renderers
/// still address absolute coordinates and see the full [`extent`](Self::extent),
/// but `cells` backs only the slab and puts outside `[z0, z1)` are dropped. The
/// full-window [`new`](Self::new) constructor sets `z0 = 0, z1 = ez`, so its
/// index reduces to the classic `x + y*ex + z*ex*ey`.
pub struct VoxelGridMut<'a> {
    cells: &'a mut [VoxelCell],
    extent: [u32; 3],
    z0: u32,
    z1: u32,
}

impl<'a> VoxelGridMut<'a> {
    /// Full-grid view: `cells` covers the whole `extent` (`z0 = 0, z1 = ez`).
    pub fn new(cells: &'a mut [VoxelCell], extent: [u32; 3]) -> Self {
        Self { cells, extent, z0: 0, z1: extent[2] }
    }

    /// Slab view: `cells` is `extent.x * extent.y * (z1 - z0)` long; an absolute
    /// z in `[z0, z1)` maps to plane `z - z0` in the buffer. `extent()` still
    /// reports the FULL box so a renderer's absolute-coordinate math is unchanged.
    pub fn slab(cells: &'a mut [VoxelCell], extent: [u32; 3], z0: u32, z1: u32) -> Self {
        Self { cells, extent, z0, z1 }
    }

    /// The full box dimensions `[x, y, z]` in voxels (not the slab depth).
    pub fn extent(&self) -> [u32; 3] {
        self.extent
    }

    /// Write one voxel (absolute coords). Coordinates outside the box — or
    /// outside this view's `[z0, z1)` slab window — are ignored.
    pub fn put(&mut self, x: u32, y: u32, z: u32, c: VoxelCell) {
        let [ex, ey, _] = self.extent;
        if x < ex && y < ey && z >= self.z0 && z < self.z1 {
            let (ex, ey) = (ex as usize, ey as usize);
            let zl = (z - self.z0) as usize;
            self.cells[x as usize + y as usize * ex + zl * ex * ey] = c;
        }
    }
}

/// What a [`VoxelRenderer::render`] call gets: one entity, its (already-fetched)
/// byte span, the grid box dimensions, and the diff-mode flag.
pub struct VoxelRenderCtx<'a> {
    pub entity: &'a VolumeEntity,
    /// The entity's bytes — `[byte_start, byte_start + byte_len)` of its source,
    /// fetched by arbvis before dispatch. The renderer decodes/samples within.
    pub bytes: &'a [u8],
    /// The full grid box `[x, y, z]` the entity's `bbox` lives inside.
    pub extent: [u32; 3],
    pub diff_mode: bool,
}

/// 3D analog of [`crate::LeafRenderer`]: decode + aggregate + colormap one
/// entity into its voxel box. Registered by id in a [`VoxelRegistry`]; the
/// entity's `renderer_id` selects which renderer runs (falling back to the
/// shape's own id).
pub trait VoxelRenderer: Send + Sync {
    fn id(&self) -> &'static str;
    fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>);

    /// Render only the voxels whose absolute z is in `z_range`, into a slab
    /// `grid` (its [`extent`](VoxelGridMut::extent) is still the FULL box, but
    /// puts outside the slab's z-window are dropped). The default re-runs the
    /// full [`render`](VoxelRenderer::render) and lets the slab grid discard the
    /// out-of-range puts — correct, but it re-decodes the entity once per slab
    /// it intersects. A renderer whose element→voxel z-mapping is cheap to
    /// intersect should override this to iterate only the in-window elements and
    /// decode each exactly once. The driver only calls this for slabs the
    /// entity's `bbox` intersects, so a renderer must confine its writes to its
    /// declared `bbox` (already the contract used for picking/manifest).
    fn render_window(
        &self,
        ctx: &VoxelRenderCtx<'_>,
        grid: &mut VoxelGridMut<'_>,
        _z_range: Range<u32>,
    ) {
        self.render(ctx, grid);
    }

    /// Reserved extension point. Historically the relative weight (e.g. element
    /// count) used to divide a streamed point-octree budget across entities;
    /// the point cloud has since been removed, so arbvis no longer calls this.
    /// Kept (with a no-op default) so downstream renderers that still override
    /// it — e.g. modelweightvis's arch view — keep compiling. Safe to drop in a
    /// coordinated change with those consumers.
    fn point_weight(&self, _ctx: &VoxelRenderCtx<'_>) -> u64 {
        0
    }

    /// Reserved extension point, paired with
    /// [`point_weight`](VoxelRenderer::point_weight). Historically emitted the
    /// per-element points for the streamed LOD octree; that view is gone, so
    /// arbvis no longer calls this. Kept (no-op default) for the same
    /// source-compatibility reason as `point_weight`.
    fn render_points(
        &self,
        _ctx: &VoxelRenderCtx<'_>,
        _budget: u64,
        _emit: &mut dyn FnMut([f32; 3], [u8; 4]),
    ) {
    }
}

/// Id-keyed registry of [`VoxelRenderer`]s, mirroring
/// [`crate::LeafRegistry`]'s renderer half. The byte-Hilbert floor needs no
/// entry (it runs the legacy whole-stream aggregation, not the entity path), so
/// [`with_defaults`](VoxelRegistry::with_defaults) is empty; a downstream
/// registers its own (e.g. `"arch"`).
#[derive(Default, Clone)]
pub struct VoxelRegistry {
    renderers: HashMap<&'static str, Arc<dyn VoxelRenderer>>,
}

impl VoxelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// No built-in voxel renderer: the byte floor colors in-shader via the LUT,
    /// so it never dispatches through here.
    pub fn with_defaults() -> Self {
        Self::new()
    }

    pub fn register_renderer(&mut self, r: Arc<dyn VoxelRenderer>) {
        self.renderers.insert(r.id(), r);
    }

    pub fn renderer(&self, id: &str) -> Option<Arc<dyn VoxelRenderer>> {
        self.renderers.get(id).cloned()
    }
}
