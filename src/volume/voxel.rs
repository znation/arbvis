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
pub struct VoxelGridMut<'a> {
    cells: &'a mut [VoxelCell],
    extent: [u32; 3],
}

impl<'a> VoxelGridMut<'a> {
    pub fn new(cells: &'a mut [VoxelCell], extent: [u32; 3]) -> Self {
        Self { cells, extent }
    }

    /// The box dimensions `[x, y, z]` in voxels.
    pub fn extent(&self) -> [u32; 3] {
        self.extent
    }

    /// Write one voxel. Coordinates outside the box are ignored.
    pub fn put(&mut self, x: u32, y: u32, z: u32, c: VoxelCell) {
        let [ex, ey, ez] = self.extent;
        if x < ex && y < ey && z < ez {
            let (ex, ey) = (ex as usize, ey as usize);
            self.cells[x as usize + y as usize * ex + z as usize * ex * ey] = c;
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
