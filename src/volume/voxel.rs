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
#[derive(Clone, Copy, Default, Debug, PartialEq)]
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
        Self {
            cells,
            extent,
            z0: 0,
            z1: extent[2],
        }
    }

    /// Slab view: `cells` is `extent.x * extent.y * (z1 - z0)` long; an absolute
    /// z in `[z0, z1)` maps to plane `z - z0` in the buffer. `extent()` still
    /// reports the FULL box so a renderer's absolute-coordinate math is unchanged.
    pub fn slab(cells: &'a mut [VoxelCell], extent: [u32; 3], z0: u32, z1: u32) -> Self {
        Self {
            cells,
            extent,
            z0,
            z1,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::shape::VoxelBox;
    use std::sync::Mutex;

    fn cell(v: u8) -> VoxelCell {
        VoxelCell {
            r: v,
            g: v,
            b: v,
            a: v,
        }
    }

    fn entity() -> VolumeEntity {
        VolumeEntity {
            source_idx: 0,
            byte_start: 0,
            byte_len: 4,
            bbox: VoxelBox {
                x0: 0,
                y0: 0,
                z0: 0,
                x1: 2,
                y1: 2,
                z1: 2,
            },
            renderer_id: "test",
            extra: Box::new(()),
        }
    }

    #[test]
    fn full_window_put_uses_x_fastest_index() {
        let mut cells = vec![VoxelCell::default(); 2 * 3 * 2];
        {
            let mut grid = VoxelGridMut::new(&mut cells, [2, 3, 2]);
            grid.put(1, 0, 0, cell(1));
            grid.put(0, 1, 0, cell(2));
            grid.put(0, 0, 1, cell(3));
            assert_eq!(grid.extent(), [2, 3, 2]);
        }
        assert_eq!(cells[1], cell(1)); // x + y*ex + z*ex*ey
        assert_eq!(cells[2], cell(2));
        assert_eq!(cells[6], cell(3));
    }

    #[test]
    fn slab_view_reports_full_extent_and_offsets_z() {
        let mut cells = vec![VoxelCell::default(); 2 * 2];
        {
            let mut grid = VoxelGridMut::slab(&mut cells, [2, 2, 3], 1, 2);
            assert_eq!(grid.extent(), [2, 2, 3]);
            grid.put(1, 1, 1, cell(7)); // absolute z=1 -> plane 0
        }
        assert_eq!(cells[3], cell(7));
    }

    #[test]
    fn slab_view_drops_puts_outside_its_z_window() {
        let mut cells = vec![VoxelCell::default(); 2 * 2];
        VoxelGridMut::slab(&mut cells, [2, 2, 3], 1, 2).put(0, 0, 0, cell(1)); // below window
        VoxelGridMut::slab(&mut cells, [2, 2, 3], 1, 2).put(0, 0, 2, cell(2)); // above window
        assert!(cells.iter().all(|c| *c == VoxelCell::default()));
    }

    #[test]
    fn out_of_range_coordinates_are_dropped_without_panic() {
        let mut cells = vec![VoxelCell::default(); 8];
        let mut grid = VoxelGridMut::new(&mut cells, [2, 2, 2]);
        grid.put(2, 0, 0, cell(1)); // x == ex
        grid.put(0, 2, 0, cell(2)); // y == ey
        grid.put(0, 0, 2, cell(3)); // z == ez
        assert!(cells.iter().all(|c| *c == VoxelCell::default()));
    }

    struct RecordingRenderer(Mutex<Vec<Vec<u8>>>);

    impl VoxelRenderer for RecordingRenderer {
        fn id(&self) -> &'static str {
            "test"
        }
        fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>) {
            self.0.lock().unwrap().push(ctx.bytes.to_vec());
            for x in 0..2 {
                grid.put(x, 0, 0, cell(9));
            }
        }
    }

    #[test]
    fn render_window_default_delegates_to_render() {
        let r = RecordingRenderer(Mutex::new(Vec::new()));
        let mut cells = vec![VoxelCell::default(); 4];
        let e = entity();
        let bytes = [1u8, 2, 3, 4];
        let ctx = VoxelRenderCtx {
            entity: &e,
            bytes: &bytes,
            extent: [2, 2, 2],
            diff_mode: false,
        };
        let mut grid = VoxelGridMut::new(&mut cells, [2, 2, 2]);
        r.render_window(&ctx, &mut grid, 0..2);
        assert_eq!(r.0.lock().unwrap().as_slice(), &[bytes.to_vec()][..]);
        assert_eq!(cells[0], cell(9));
        assert_eq!(cells[1], cell(9));
    }

    #[test]
    fn registry_looks_up_by_id_and_misses_unknown() {
        let mut reg = VoxelRegistry::with_defaults();
        assert!(reg.renderer("test").is_none());
        reg.register_renderer(Arc::new(RecordingRenderer(Mutex::new(Vec::new()))));
        assert!(reg.renderer("test").is_some());
        assert!(reg.renderer("other").is_none());
    }
}
