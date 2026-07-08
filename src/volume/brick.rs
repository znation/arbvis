//! Sparse-voxel **brick pool + page table** for the volume ray-march — the
//! GigaVoxels-style indirection that lets the viewer store and sample only the
//! *occupied* regions of the volume instead of one dense `Data3DTexture`.
//!
//! The dense grid wastes storage (and, at high resolution, GPU texture budget)
//! on the empty majority of the cube — the Hilbert prefix fills only part of
//! it, and structured layouts have gaps. Here the volume is diced into `B³`
//! **bricks**; only bricks containing an occupied voxel (`a > 0`) are kept and
//! packed into a 3D **brick-pool atlas**. A coarse **page table** (one cell per
//! brick) maps each brick position to its atlas slot, or marks it empty. The
//! ray-march reads the page table to leap empty bricks and to find the atlas
//! texels for occupied ones.
//!
//! Two builders share the format:
//! - [`build_brick_volume`] transforms a finished dense grid — bounded by the
//!   dense resolution, used by the structured path and the byte default.
//! - [`BrickBuilder`] streams bricks straight off the byte→Hilbert pass at a
//!   higher virtual resolution to *exceed* the dense grid. Because an aligned
//!   `B³` brick is a contiguous Hilbert range, bricks finalize one at a time as
//!   the curve advances, so the accumulator is `O(one brick)` regardless of
//!   resolution — and each finished brick is written straight to disk, so RAM
//!   never holds the atlas (only the `O(occupied)` octree grows with the data).
//! - [`StreamBrickAgg`] is the dense/structured analog: it bricks brick-aligned
//!   Z-slabs of a baked grid, likewise streaming bricks to disk as it goes.

use std::io::Write;

use super::encode::VoxelAcc;
use crate::geometry::{hilbert3d_node_origin, hilbert_d2xyz};

/// Brick edge in voxels. `8³ = 512` voxels = 2 KiB RGBA8 per brick — a good
/// granularity for both empty-space skipping and the atlas.
pub const BRICK: u32 = 8;

/// 1-voxel border replicated around each dense-derived brick so the viewer can
/// trilinearly filter *across* brick boundaries (sampling the apron instead of
/// bleeding into an unrelated atlas slot). The streaming builder can't see a
/// brick's neighbors, so it ships `apron = 0` (nearest filtering).
pub const APRON: u32 = 1;

/// Marks an octree entry as a **leaf** (a brick), as opposed to an internal
/// child-node pointer. Set on the packed entry during building so the serializer
/// can tell the two apart; the low bits carry the 1-based brick id. Both brick
/// ids and node indices stay well under 2³¹, so the top bit is free.
const LEAF_BIT: u32 = 1 << 31;

/// Near-cubic atlas size, in bricks per axis, holding at least `slots` bricks.
fn atlas_dims_bricks(slots: u32) -> [u32; 3] {
    let s = slots.max(1);
    let ax = (s as f64).cbrt().ceil() as u32;
    let ax = ax.max(1);
    let ay = ax;
    let az = s.div_ceil(ax * ay).max(1);
    [ax, ay, az]
}

/// Pack a dense page table (`0` = empty, else 1-based slot) into RGBA8 with the
/// slot encoded little-endian in R,G,B (A = 255).
fn pack_page_table(slots: &[u32]) -> Vec<u8> {
    let mut out = vec![0u8; slots.len() * 4];
    for (i, &slot) in slots.iter().enumerate() {
        out[i * 4] = (slot & 0xff) as u8;
        out[i * 4 + 1] = ((slot >> 8) & 0xff) as u8;
        out[i * 4 + 2] = ((slot >> 16) & 0xff) as u8;
        out[i * 4 + 3] = 255;
    }
    out
}

/// Sparse octree (N³-tree) over a `2^depth`³ **cube** of brick cells, built by
/// insertion — `O(occupied)`, never the dense `(2^depth)³` cube. `nodes[0]` is
/// the root; each node is 8 child entries (octant `x | y<<1 | z<<2`): `0` = empty
/// subtree, `LEAF_BIT | id` = a 1-based brick leaf, else a 1-based child-node
/// index. Shared by the byte→Hilbert [`BrickBuilder`] and the dense-RGBA
/// [`build_streamed_brick_volume`]; the descent (MSB-first octant bits) matches
/// the shader's boundary-compare walk and the test `descend_octree`.
struct Octree {
    nodes: Vec<[u32; 8]>,
    depth: u32,
}

impl Octree {
    fn new(depth: u32) -> Self {
        Octree { nodes: vec![[0u32; 8]], depth }
    }

    /// Insert an occupied brick at cell `(cx, cy, cz)` (in brick coords) with the
    /// 1-based `id`, creating internal nodes along the path as needed. Descends
    /// `depth` levels; at each level the octant bit is taken from the coordinate
    /// bit `depth-1-d`, so the shader's boundary-compare descent visits the same
    /// child. The deepest level stores the leaf entry (`LEAF_BIT | id`).
    fn insert(&mut self, cell: [u32; 3], id: u32) {
        let mut node = 0usize;
        for d in 0..self.depth {
            let shift = self.depth - 1 - d;
            let octant = (((cell[0] >> shift) & 1)
                | (((cell[1] >> shift) & 1) << 1)
                | (((cell[2] >> shift) & 1) << 2)) as usize;
            if d == self.depth - 1 {
                self.nodes[node][octant] = LEAF_BIT | id;
            } else {
                let child = self.nodes[node][octant];
                node = if child == 0 {
                    self.nodes.push([0u32; 8]);
                    let ni = (self.nodes.len() - 1) as u32; // 0-based
                    self.nodes[node][octant] = ni + 1; // store 1-based
                    ni as usize
                } else {
                    (child - 1) as usize
                };
            }
        }
    }

    /// Pack the octree into an RGBA8 node-pool texture: each node is a 2×2×2
    /// block of child entries, nodes laid out near-cubically (like the brick
    /// atlas). Leaf entries carry `A = 1` (occupied, not resident) + RGB brick
    /// id; internal entries carry `A = 0` + RGB 1-based child-node index; empty
    /// entries stay all-zero. Returns `(bytes, dims_in_texels, node_count)`.
    fn serialize(&self) -> (Vec<u8>, [u32; 3], u32) {
        let n = self.nodes.len() as u32;
        let nx = ((n as f64).cbrt().ceil() as u32).max(1);
        let ny = nx;
        let nz = n.div_ceil(nx * ny).max(1);
        let (tw, th, td) = (nx * 2, ny * 2, nz * 2);
        let mut out = vec![0u8; (tw as usize) * (th as usize) * (td as usize) * 4];
        for (ni, node) in self.nodes.iter().enumerate() {
            let ni = ni as u32;
            let (bx, by, bz) = (ni % nx, (ni / nx) % ny, ni / (nx * ny));
            for (octant, &e) in node.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let (ox, oy, oz) = (octant as u32 & 1, (octant as u32 >> 1) & 1, (octant as u32 >> 2) & 1);
                let (tx, ty, tz) = (bx * 2 + ox, by * 2 + oy, bz * 2 + oz);
                let ti = ((tx + ty * tw + tz * tw * th) as usize) * 4;
                let (payload, state) = if e & LEAF_BIT != 0 {
                    (e & !LEAF_BIT, 1u8) // leaf: brick id, occupied-not-resident
                } else {
                    (e, 0u8) // internal: 1-based child-node index
                };
                out[ti] = (payload & 0xff) as u8;
                out[ti + 1] = ((payload >> 8) & 0xff) as u8;
                out[ti + 2] = ((payload >> 16) & 0xff) as u8;
                out[ti + 3] = state;
            }
        }
        (out, [tw, th, td], n)
    }
}

/// The packed sparse volume: a brick-pool atlas + a page table indexing it.
pub struct BrickVolume {
    /// Brick-pool atlas, RGBA8, `atlas_dim` voxels (x-fastest). Occupied bricks
    /// are packed slot-by-slot; unused atlas space is left transparent. **Empty
    /// for the streamed builders** ([`StreamBrickAgg`], [`BrickBuilder`]): they
    /// write each brick straight to `bricks.bin` as it finalizes, so the atlas
    /// never accumulates in RAM and the caller does not re-write it.
    pub atlas: Vec<u8>,
    /// Atlas dimensions in voxels `[x, y, z]` (each a multiple of [`BRICK`]).
    pub atlas_dim: [u32; 3],
    /// Page table, RGBA8, `page_dim` cells (x-fastest): the 1-based atlas slot
    /// of each brick encoded little-endian in R,G,B (`0` ⇒ empty brick). A is
    /// unused (255). The viewer samples this (nearest) to route each step.
    pub page_table: Vec<u8>,
    /// Page-table dimensions in bricks `[x, y, z]` (`ceil(extent / BRICK)`).
    pub page_dim: [u32; 3],
    /// Voxel extent the page table represents `[x, y, z]` — the dense grid
    /// extent for [`build_brick_volume`], or the virtual side for the
    /// high-resolution [`BrickBuilder`]. The viewer maps `uvw·vol_dim → voxel`,
    /// so this (not the dense `grid_extent`) drives brick addressing.
    pub vol_dim: [u32; 3],
    /// Apron border per brick: stored brick edge is `BRICK + 2·apron`. `1` ⇒ the
    /// viewer trilinearly filters across brick edges; `0` ⇒ nearest.
    pub apron: u32,
    /// Number of occupied bricks (atlas slots used).
    pub occupied: u32,
    /// Largest raw per-voxel byte count in the streamed byte atlas (density
    /// source). `> 0` ⇒ the atlas B channel holds RAW counts and the viewer
    /// normalizes density by `255/max_count` when sampling a resident brick;
    /// `0` ⇒ B is already baked/normalized (structured verbatim RGBA and the
    /// non-streamed path), so the viewer does not rescale.
    pub max_count: u64,
    /// `false` ⇒ `atlas` is a packed 3D atlas (`atlas_dim` voxels) the viewer
    /// uploads whole, and the page table holds resident atlas slots. `true` ⇒
    /// `atlas` is a flat, range-addressable array of `occupied` bricks
    /// (`BRICK³·4` bytes each, brick `S` at `(S-1)·BRICK³·4`), `atlas_dim` is
    /// unused, and instead of `page_table` the sparse **octree** `node_pool`
    /// indexes bricks; the viewer ray-guides bricks into a bounded GPU cache on
    /// demand.
    pub streamed: bool,

    // --- Streamed-path sparse octree (empty unless `streamed`). Replaces the
    // flat O((vol/BRICK)³) page table with an O(occupied) N³-tree so the page
    // structure — download, client RAM, and VRAM — no longer scales with the
    // volume. See [`BrickBuilder`].
    /// Octree node pool, RGBA8, `node_pool_dim` texels (x-fastest). Each node is
    /// a 2×2×2 block of child entries; entry `A>0` ⇒ a leaf (brick: RGB = 1-based
    /// brick id on disk, `A = 1` occupied-not-resident), `A==0 && RGB>0` ⇒ an
    /// internal pointer (RGB = 1-based child-node index), all-zero ⇒ empty
    /// subtree. Node 0 is the root, covering the whole `2^tree_depth` brick cube.
    pub node_pool: Vec<u8>,
    /// Node-pool texture dims in texels `[x, y, z]` (each even: nodes packed
    /// near-cubically, 2 texels/node/axis).
    pub node_pool_dim: [u32; 3],
    /// Octree depth `D = log2(bricks per side)`; the descent walks at most `D`
    /// levels. `0` ⇒ no octree (the flat `page_table` is used instead).
    pub tree_depth: u32,
    /// Number of octree nodes (root + internal), for logging/meta.
    pub node_count: u32,
    /// World-space camera framing (center, radius) of the **fine** occupied
    /// region. The streamed path must frame this, not the coarse grid: for a
    /// small file at high resolution the coarse grid fills the whole cube while
    /// the fine data is a tiny Hilbert-prefix corner, so coarse framing would
    /// point the camera at empty space. `(origin, 0.5)` for the non-streamed path.
    pub focus_center: [f32; 3],
    pub focus_radius: f32,
}

impl BrickVolume {
    /// Atlas size in bricks per axis (`atlas_dim / (BRICK + 2·apron)`). The
    /// viewer derives this in JS; here it backs the format tests.
    #[cfg(test)]
    pub fn atlas_bricks(&self) -> [u32; 3] {
        let bs = BRICK + 2 * self.apron;
        [self.atlas_dim[0] / bs, self.atlas_dim[1] / bs, self.atlas_dim[2] / bs]
    }
}

/// Build a [`BrickVolume`] from a finished RGBA8 dense grid (`extent` voxels).
/// Bricks with no occupied voxel (`a > 0`) are dropped; the rest are packed
/// into a near-cubic atlas. Works for both the byte (`lut`) and structured
/// (`rgb`) paths since both encode emptiness as `a == 0`.
pub fn build_brick_volume(rgba: &[u8], extent: [u32; 3], brick: u32) -> BrickVolume {
    let [ex, ey, ez] = extent;
    let (pbx, pby, pbz) = (ex.div_ceil(brick), ey.div_ceil(brick), ez.div_ceil(brick));
    let npages = (pbx as usize) * (pby as usize) * (pbz as usize);
    let (exs, eys) = (ex as usize, ey as usize);
    let page_idx = |bx: u32, by: u32, bz: u32| -> usize {
        (bz as usize) * (pby as usize) * (pbx as usize)
            + (by as usize) * (pbx as usize)
            + bx as usize
    };

    // Pass 1: which bricks are occupied?
    let mut occ = vec![false; npages];
    for z in 0..ez {
        for y in 0..ey {
            for x in 0..ex {
                if rgba[(x as usize + y as usize * exs + z as usize * exs * eys) * 4 + 3] > 0 {
                    occ[page_idx(x / brick, y / brick, z / brick)] = true;
                }
            }
        }
    }

    // Assign 1-based atlas slots to occupied bricks (0 stays "empty").
    let mut slot_of = vec![0u32; npages];
    let mut occupied = 0u32;
    for (i, &o) in occ.iter().enumerate() {
        if o {
            occupied += 1;
            slot_of[i] = occupied; // 1-based; the viewer subtracts 1
        }
    }

    // Near-cubic atlas big enough for every slot. Each stored brick carries an
    // APRON-voxel border so the viewer can filter across brick edges; the
    // stored brick edge is `bs`.
    let bs = brick + 2 * APRON;
    let [ax, ay, az] = atlas_dims_bricks(occupied);
    let atlas_dim = [ax * bs, ay * bs, az * bs];
    let (adx, ady) = (atlas_dim[0] as usize, atlas_dim[1] as usize);
    let mut atlas = vec![0u8; (atlas_dim[0] as usize) * (atlas_dim[1] as usize) * (atlas_dim[2] as usize) * 4];

    // Pass 2: fill each occupied brick's slot from the dense grid, including the
    // apron border (neighbor voxels, or transparent past the grid edge / into
    // empty neighbors).
    let a = APRON as i64;
    for bz in 0..pbz {
        for by in 0..pby {
            for bx in 0..pbx {
                let slot = slot_of[page_idx(bx, by, bz)];
                if slot == 0 {
                    continue;
                }
                let s = slot - 1;
                let (sx, sy, sz) = (s % ax, (s / ax) % ay, s / (ax * ay));
                for dz in 0..bs {
                    for dy in 0..bs {
                        for dx in 0..bs {
                            let gx = bx as i64 * brick as i64 + dx as i64 - a;
                            let gy = by as i64 * brick as i64 + dy as i64 - a;
                            let gz = bz as i64 * brick as i64 + dz as i64 - a;
                            if gx < 0
                                || gy < 0
                                || gz < 0
                                || gx as u32 >= ex
                                || gy as u32 >= ey
                                || gz as u32 >= ez
                            {
                                continue; // border past grid edge → transparent
                            }
                            let src =
                                (gx as usize + gy as usize * exs + gz as usize * exs * eys) * 4;
                            let dst = ((sx * bs + dx) as usize
                                + (sy * bs + dy) as usize * adx
                                + (sz * bs + dz) as usize * adx * ady)
                                * 4;
                            atlas[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                        }
                    }
                }
            }
        }
    }

    BrickVolume {
        atlas,
        atlas_dim,
        page_table: pack_page_table(&slot_of),
        page_dim: [pbx, pby, pbz],
        vol_dim: extent,
        apron: APRON,
        occupied,
        max_count: 0, // non-streamed atlas keeps B baked; viewer never rescales
        streamed: false,
        // Non-streamed uses the flat page table above, not the octree; the
        // caller derives framing from the dense grid instead.
        node_pool: Vec::new(),
        node_pool_dim: [0, 0, 0],
        tree_depth: 0,
        node_count: 0,
        focus_center: [0.0, 0.0, 0.0],
        focus_radius: 0.5,
    }
}

/// Streaming brick aggregator for the dense/structured (`rgb`) path — the
/// dense-grid analog of [`BrickBuilder`], for layouts that bake final color
/// rather than streaming bytes in Hilbert order. It bricks a sequence of
/// **brick-aligned Z-slabs** into the same on-the-wire shape as the byte
/// streamed path (a flat, range-addressable block array + a sparse octree over
/// a `2^depth`³ **cube** of brick cells), writing each occupied brick straight
/// to a [`Write`] sink (`bricks.bin`) as it finalizes — so the atlas never
/// accumulates in RAM.
///
/// Because slabs advance in increasing z and each slab bricks in `bz/by/bx`
/// order, brick ids stay sequential (brick `S` at byte `(S-1)·brick³·4`) and
/// octree insertion matches a single full-grid pass; a single full-extent slab
/// reproduces the old one-shot builder exactly. Bricks store RGBA **verbatim**
/// (no mean/luma/density transform), so the streamed+`directColor` shader
/// renders them unchanged, and the full anisotropic `extent` is the `vol_dim`.
/// `apron` is `0` (nearest filtering): the flat block layout can't carry
/// neighbour borders. Framing uses the occupied-brick bbox via
/// [`super::box_focus`].
pub struct StreamBrickAgg {
    extent: [u32; 3],
    brick: u32,
    octree: Octree,
    occupied: u32,
    // Occupied-brick voxel bbox + centroid, for fine-data camera framing.
    fmin: [u32; 3],
    fmax: [u32; 3],
    fsum: [f64; 3],
}

impl StreamBrickAgg {
    pub fn new(extent: [u32; 3], brick: u32) -> Self {
        // Page dims per axis (bricks), and the power-of-two brick cube the octree
        // spans. Short axes never populate their high cells → empty subtrees.
        let pb = [
            extent[0].div_ceil(brick),
            extent[1].div_ceil(brick),
            extent[2].div_ceil(brick),
        ];
        let p = pb[0].max(pb[1]).max(pb[2]).next_power_of_two().max(2);
        StreamBrickAgg {
            extent,
            brick,
            octree: Octree::new(p.trailing_zeros()),
            occupied: 0,
            fmin: [u32::MAX; 3],
            fmax: [0; 3],
            fsum: [0.0; 3],
        }
    }

    /// Brick one **brick-aligned** Z-slab (`z0 % brick == 0`; `z1` a multiple of
    /// `brick`, or `== extent.z` for the final slab). `slab` is `ex·ey·(z1-z0)`
    /// RGBA8 (x-fastest, planes z-relative to `z0`). Emits every occupied brick
    /// in the slab (absolute `bz/by/bx` order) to `writer` and inserts it into
    /// the octree. Slab boundaries being `brick` multiples guarantees no brick
    /// row straddles two slabs.
    pub fn add_slab<W: Write>(
        &mut self,
        slab: &[u8],
        z0: u32,
        z1: u32,
        writer: &mut W,
    ) -> std::io::Result<()> {
        debug_assert!(z0.is_multiple_of(self.brick), "slab boundary must be brick-aligned");
        let [ex, ey, _] = self.extent;
        let (exs, eys) = (ex as usize, ey as usize);
        let brick = self.brick;
        let bk = brick as usize;
        let mut scratch = vec![0u8; bk * bk * bk * 4];
        for bz in (z0 / brick)..z1.div_ceil(brick) {
            for by in 0..ey.div_ceil(brick) {
                for bx in 0..ex.div_ceil(brick) {
                    scratch.iter_mut().for_each(|v| *v = 0);
                    let mut any = false;
                    for dz in 0..brick {
                        let gz = bz * brick + dz;
                        if gz >= z1 {
                            break; // clip to the slab (or the final partial brick)
                        }
                        let zl = (gz - z0) as usize; // slab-relative plane
                        for dy in 0..brick {
                            let gy = by * brick + dy;
                            if gy >= ey {
                                break;
                            }
                            for dx in 0..brick {
                                let gx = bx * brick + dx;
                                if gx >= ex {
                                    break;
                                }
                                let src = (gx as usize + gy as usize * exs + zl * exs * eys) * 4;
                                let dst =
                                    (dx as usize + dy as usize * bk + dz as usize * bk * bk) * 4;
                                scratch[dst..dst + 4].copy_from_slice(&slab[src..src + 4]);
                                any |= slab[src + 3] > 0;
                            }
                        }
                    }
                    if !any {
                        continue;
                    }
                    self.occupied += 1; // 1-based brick id
                    self.octree.insert([bx, by, bz], self.occupied);
                    writer.write_all(&scratch)?;
                    // Track the occupied region in voxels (brick origin → +brick-1).
                    let origin = [bx * brick, by * brick, bz * brick];
                    for (a, &o) in origin.iter().enumerate() {
                        self.fmin[a] = self.fmin[a].min(o);
                        self.fmax[a] = self.fmax[a].max(o + brick - 1);
                        self.fsum[a] += (o + brick / 2) as f64;
                    }
                }
            }
        }
        Ok(())
    }

    /// Finalize the octree + framing into a streamed [`BrickVolume`]. Its `atlas`
    /// is empty — every brick was already written to the `Write` sink.
    pub fn finish(self) -> BrickVolume {
        let pb = [
            self.extent[0].div_ceil(self.brick),
            self.extent[1].div_ceil(self.brick),
            self.extent[2].div_ceil(self.brick),
        ];
        let (node_pool, node_pool_dim, node_count) = self.octree.serialize();
        let (focus_center, focus_radius) = super::box_focus(
            if self.occupied > 0 { self.fmin } else { [0; 3] },
            self.fmax,
            self.fsum,
            self.occupied as u64,
            self.extent,
        );
        BrickVolume {
            atlas: Vec::new(),
            atlas_dim: [0, 0, 0], // streamed: bricks.bin is a flat block array
            page_table: Vec::new(), // streamed uses the octree node pool instead
            page_dim: pb,
            vol_dim: self.extent,
            apron: 0, // streaming can't see neighbors → no border, nearest filtering
            occupied: self.occupied,
            max_count: 0, // structured RGBA is verbatim; viewer does not rescale
            streamed: true,
            node_pool,
            node_pool_dim,
            tree_depth: self.octree.depth,
            node_count,
            focus_center,
            focus_radius,
        }
    }
}

/// Build a **streamed** [`BrickVolume`] from a finished RGBA8 dense grid in one
/// shot (a single full-extent slab), writing bricks to `writer`. A test/helper
/// wrapper over [`StreamBrickAgg`]; the production structured path drives the
/// aggregator slab-by-slab so it never materializes the full dense grid.
#[cfg(test)]
pub fn build_streamed_brick_volume<W: Write>(
    rgba: &[u8],
    extent: [u32; 3],
    brick: u32,
    writer: &mut W,
) -> std::io::Result<BrickVolume> {
    let mut agg = StreamBrickAgg::new(extent, brick);
    agg.add_slab(rgba, 0, extent[2], writer)?;
    Ok(agg.finish())
}

/// Streaming sparse-voxel brick builder for the byte→Hilbert pass. Points are
/// fed in non-decreasing Hilbert order at virtual order `order_v` (cube side
/// `2^order_v`); because an aligned `2^brick_log2` brick is a contiguous Hilbert
/// range, only the *current* brick is open at a time. Finished bricks are
/// written straight to the [`Write`] sink (`bricks.bin`) as they finalize, and a
/// sparse octree indexes them; [`finish_streaming`](BrickBuilder::finish_streaming)
/// assembles the [`BrickVolume`]. Memory is `O(one brick)` plus the O(occupied)
/// octree — never the full `2^order_v` cube, nor the atlas (it lives on disk).
pub struct BrickBuilder<W: Write> {
    order_v: u32,
    brick: u32,
    brick_log2: u32,
    luma: [u16; 256],
    page_dim: [u32; 3], // bricks per axis: [P, P, P], P = 2^depth
    // Sparse octree over the P³ brick grid (P = 2^depth = 2^(order_v-brick_log2)),
    // built by insertion as bricks finalize (O(occupied) memory, never the P³ cube).
    octree: Octree,
    writer: W,          // occupied bricks stream here (bricks.bin) as they finalize
    brick_buf: Vec<u8>, // reusable B³·4-byte staging buffer for one brick
    io_err: Option<std::io::Error>, // first write error, surfaced at finish_streaming
    open: Vec<VoxelAcc>, // accumulators for the current open brick (B³)
    cur_hidx: i128,    // Hilbert index of the open brick (-1 = none)
    cur_origin: [u32; 3],
    occupied: u32,
    max_count: u64,
    // Voxel-space bounding box + centroid of occupied bricks, for framing the
    // camera on the *fine* data (see BrickVolume::focus_center).
    fmin: [u32; 3],
    fmax: [u32; 3],
    fsum: [f64; 3],
}

impl<W: Write> BrickBuilder<W> {
    /// `order_v` = virtual cube exponent (side `2^order_v`); `brick` = brick
    /// edge (power of two ≤ `2^order_v`); `luma` = per-byte luminance LUT;
    /// `writer` = sink for the flat brick blocks (`bricks.bin`).
    pub fn new(order_v: u32, brick: u32, luma: [u16; 256], writer: W) -> Self {
        assert!((1..=21).contains(&order_v));
        let brick_log2 = brick.trailing_zeros();
        let side = 1u32 << order_v;
        let pd = side / brick;
        let depth = order_v - brick_log2; // P = 2^depth bricks per side
        assert!(depth >= 1, "streamed octree needs ≥ 2 bricks per side");
        BrickBuilder {
            order_v,
            brick,
            brick_log2,
            luma,
            page_dim: [pd, pd, pd],
            octree: Octree::new(depth),
            writer,
            brick_buf: vec![0u8; (brick as usize).pow(3) * 4],
            io_err: None,
            open: vec![VoxelAcc::default(); (brick as usize).pow(3)],
            cur_hidx: -1,
            cur_origin: [0; 3],
            occupied: 0,
            max_count: 0,
            fmin: [u32::MAX; 3],
            fmax: [0; 3],
            fsum: [0.0; 3],
        }
    }

    /// Accumulate one byte `b` whose voxel is at Hilbert distance `h` on the
    /// `2^order_v` cube. `h` must be non-decreasing across calls.
    pub fn push(&mut self, h: u64, b: u8) {
        let hidx = (h >> (3 * self.brick_log2)) as i128;
        if hidx != self.cur_hidx {
            self.finalize();
            self.cur_hidx = hidx;
            self.cur_origin =
                hilbert3d_node_origin(hidx as u64, self.order_v - self.brick_log2, self.order_v);
            for a in self.open.iter_mut() {
                *a = VoxelAcc::default();
            }
        }
        let v = hilbert_d2xyz(h, self.order_v);
        let bk = self.brick;
        let li = (v[0] - self.cur_origin[0])
            + (v[1] - self.cur_origin[1]) * bk
            + (v[2] - self.cur_origin[2]) * bk * bk;
        let acc = &mut self.open[li as usize];
        acc.count += 1;
        acc.sum_val += b as u64;
        acc.sum_luma += self.luma[b as usize] as u64;
    }

    /// Emit the open brick (if non-empty): stage it into `brick_buf`, insert it
    /// into the octree, and write it to the `writer` sink.
    fn finalize(&mut self) {
        if self.cur_hidx < 0 || self.open.iter().all(|a| a.count == 0) {
            return;
        }
        self.occupied += 1; // 1-based brick id
        let bk = self.brick;
        let cell = [
            self.cur_origin[0] / bk,
            self.cur_origin[1] / bk,
            self.cur_origin[2] / bk,
        ];
        // Track the occupied region (in voxels) for fine-data camera framing.
        for a in 0..3 {
            self.fmin[a] = self.fmin[a].min(self.cur_origin[a]);
            self.fmax[a] = self.fmax[a].max(self.cur_origin[a] + bk - 1);
            self.fsum[a] += (self.cur_origin[a] + bk / 2) as f64;
        }
        self.octree.insert(cell, self.occupied);
        // Stage the brick into brick_buf. B holds the RAW per-voxel count
        // (clamped to 255); the global density rescale is deferred to the shader
        // (see `BrickVolume::max_count`) because bricks stream to disk before the
        // final `max_count` is known. Empty voxels (a == 0) stay transparent.
        for i in 0..self.open.len() {
            let acc = self.open[i];
            let o = i * 4;
            if acc.count == 0 {
                self.brick_buf[o..o + 4].copy_from_slice(&[0, 0, 0, 0]);
            } else {
                let c = acc.count as u64;
                let mean = (acc.sum_val / c).min(255) as u8;
                let act = (acc.sum_luma / c).min(255) as u8;
                self.brick_buf[o..o + 4].copy_from_slice(&[mean, act, c.min(255) as u8, 255]);
                self.max_count = self.max_count.max(c);
            }
        }
        if self.io_err.is_none() {
            if let Err(e) = self.writer.write_all(&self.brick_buf) {
                self.io_err = Some(e);
            }
        }
    }

    /// Finish for the **streamed** viewer path: keep the occupied bricks as a
    /// flat, range-addressable block array — brick `S` (1-based) at byte
    /// `(S-1)·brick³·4`, `brick³·4` bytes long — instead of scattering them into
    /// a packed atlas, and ship the sparse **octree** `node_pool` indexing them.
    /// The viewer streams bricks into a bounded GPU cache on demand (ray-guided),
    /// so the full occupied set never has to be GPU-resident, and the octree
    /// keeps the page structure O(occupied) rather than O((side/brick)³) — the
    /// reason the streaming path can exceed the dense grid in both VRAM *and*
    /// download/RAM without either scaling with the volume.
    ///
    /// The per-voxel density (B) rescale is **not** applied here — the atlas
    /// ships raw counts and the viewer normalizes by `255/max_count` (carried on
    /// [`BrickVolume::max_count`]). Returns the volume plus the sink `W` (so
    /// tests can recover an in-memory buffer); a write error stashed during
    /// streaming surfaces here.
    pub fn finish_streaming(mut self) -> std::io::Result<(BrickVolume, W)> {
        self.finalize();
        if let Some(e) = self.io_err.take() {
            return Err(e);
        }
        self.writer.flush()?;
        let (node_pool, node_pool_dim, node_count) = self.octree.serialize();
        let side = 1u32 << self.order_v;
        // Frame on the fine occupied region (voxel bbox → world). box_focus
        // falls back to the whole cube when nothing is occupied.
        let (focus_center, focus_radius) = super::box_focus(
            if self.occupied > 0 { self.fmin } else { [0; 3] },
            self.fmax,
            self.fsum,
            self.occupied as u64,
            [side, side, side],
        );
        let bv = BrickVolume {
            atlas: Vec::new(), // bricks were streamed to `writer` (bricks.bin)
            atlas_dim: [0, 0, 0], // streamed: bricks.bin is a flat block array
            page_table: Vec::new(), // streamed uses the octree node pool instead
            page_dim: self.page_dim,
            vol_dim: [side, side, side],
            apron: 0, // streaming can't see neighbors → no border, nearest filtering
            occupied: self.occupied,
            max_count: self.max_count, // >0 ⇒ shader normalizes density by 255/max_count
            streamed: true,
            node_pool,
            node_pool_dim,
            tree_depth: self.octree.depth,
            node_count,
            focus_center,
            focus_radius,
        };
        Ok((bv, self.writer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a dense grid with a single occupied voxel at `(x,y,z)` colored
    /// `c`, everything else transparent.
    fn one_voxel(extent: [u32; 3], at: (u32, u32, u32), c: [u8; 4]) -> Vec<u8> {
        let [ex, ey, ez] = extent;
        let mut g = vec![0u8; (ex * ey * ez * 4) as usize];
        let i = ((at.0 + at.1 * ex + at.2 * ex * ey) * 4) as usize;
        g[i..i + 4].copy_from_slice(&c);
        g
    }

    /// Reconstruct a voxel's RGBA by walking the page table + atlas exactly as
    /// the viewer will, returning `[0,0,0,0]` for empty bricks.
    fn sample(bv: &BrickVolume, extent: [u32; 3], x: u32, y: u32, z: u32) -> [u8; 4] {
        let [pbx, pby, _] = bv.page_dim;
        let pi = ((z / BRICK) * pby * pbx + (y / BRICK) * pbx + (x / BRICK)) as usize;
        let slot = bv.page_table[pi * 4] as u32
            | (bv.page_table[pi * 4 + 1] as u32) << 8
            | (bv.page_table[pi * 4 + 2] as u32) << 16;
        if slot == 0 {
            return [0, 0, 0, 0];
        }
        let [ax, ay, _] = bv.atlas_bricks();
        let s = slot - 1;
        let (sx, sy, sz) = (s % ax, (s / ax) % ay, s / (ax * ay));
        let [adx, ady, _] = bv.atlas_dim;
        let bs = BRICK + 2 * bv.apron; // stored brick edge
        let ap = bv.apron;
        let (axx, ayy, azz) = (
            sx * bs + ap + x % BRICK,
            sy * bs + ap + y % BRICK,
            sz * bs + ap + z % BRICK,
        );
        let di = ((axx + ayy * adx + azz * adx * ady) * 4) as usize;
        let _ = extent;
        [bv.atlas[di], bv.atlas[di + 1], bv.atlas[di + 2], bv.atlas[di + 3]]
    }

    #[test]
    fn one_occupied_brick_round_trips_and_drops_the_rest() {
        let extent = [32, 32, 32]; // 4³ = 64 bricks at BRICK=8
        let g = one_voxel(extent, (3, 5, 9), [10, 20, 30, 200]);
        let bv = build_brick_volume(&g, extent, BRICK);
        assert_eq!(bv.page_dim, [4, 4, 4]);
        assert_eq!(bv.occupied, 1, "exactly one brick is occupied");
        // The occupied voxel reconstructs exactly.
        assert_eq!(sample(&bv, extent, 3, 5, 9), [10, 20, 30, 200]);
        // A voxel in a different (empty) brick is empty.
        assert_eq!(sample(&bv, extent, 31, 31, 31), [0, 0, 0, 0]);
        // A different voxel in the SAME brick is transparent (only the one set).
        assert_eq!(sample(&bv, extent, 0, 0, 8 /* brick (0,0,1) */), [0, 0, 0, 0]);
    }

    #[test]
    fn every_occupied_voxel_round_trips() {
        // A 16³ grid (8 bricks) with a scattered set of occupied voxels.
        let extent = [16, 16, 16];
        let mut g = vec![0u8; 16 * 16 * 16 * 4];
        let mut expect = Vec::new();
        let mut s: u64 = 0xC0FFEE;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..200 {
            let x = (next() % 16) as u32;
            let y = (next() % 16) as u32;
            let z = (next() % 16) as u32;
            let c = [(next() % 255 + 1) as u8, 7, 8, (next() % 200 + 1) as u8];
            let i = ((x + y * 16 + z * 16 * 16) * 4) as usize;
            g[i..i + 4].copy_from_slice(&c);
            expect.push(((x, y, z), c));
        }
        let bv = build_brick_volume(&g, extent, BRICK);
        // Atlas is big enough for the occupied bricks.
        let [ax, ay, az] = bv.atlas_bricks();
        assert!(ax * ay * az >= bv.occupied, "atlas holds every slot");
        // Every occupied voxel reconstructs (last write wins for repeats).
        for (pos, c) in expect {
            // recompute expected from grid (a later write to the same voxel wins)
            let i = ((pos.0 + pos.1 * 16 + pos.2 * 16 * 16) * 4) as usize;
            let exp = [g[i], g[i + 1], g[i + 2], g[i + 3]];
            assert_eq!(sample(&bv, extent, pos.0, pos.1, pos.2), exp, "voxel {pos:?}");
            let _ = c;
        }
    }

    #[test]
    fn empty_grid_produces_no_bricks() {
        let extent = [16, 16, 16];
        let bv = build_brick_volume(&vec![0u8; 16 * 16 * 16 * 4], extent, BRICK);
        assert_eq!(bv.occupied, 0);
        assert!(bv.page_table.iter().step_by(4).all(|&r| r == 0), "all cells empty");
    }

    /// Descend the serialized **octree** node pool exactly as the shader/JS will,
    /// returning the 1-based brick id at brick cell `(cx,cy,cz)` (`0` = empty).
    /// This is the authoritative check that the built tree round-trips.
    fn descend_octree(bv: &BrickVolume, cx: u32, cy: u32, cz: u32) -> u32 {
        let [tw, th, _] = bv.node_pool_dim;
        let (nx, ny) = (tw / 2, th / 2);
        let read = |node: u32, ox: u32, oy: u32, oz: u32| -> (u32, u8) {
            let (bx, by, bz) = (node % nx, (node / nx) % ny, node / (nx * ny));
            let (txc, tyc, tzc) = (bx * 2 + ox, by * 2 + oy, bz * 2 + oz);
            let ti = ((txc + tyc * tw + tzc * tw * th) as usize) * 4;
            let rgb = bv.node_pool[ti] as u32
                | (bv.node_pool[ti + 1] as u32) << 8
                | (bv.node_pool[ti + 2] as u32) << 16;
            (rgb, bv.node_pool[ti + 3])
        };
        let mut node = 0u32;
        for d in 0..bv.tree_depth {
            let shift = bv.tree_depth - 1 - d;
            let (ox, oy, oz) = ((cx >> shift) & 1, (cy >> shift) & 1, (cz >> shift) & 1);
            let (rgb, a) = read(node, ox, oy, oz);
            if a > 0 {
                return rgb; // leaf: 1-based brick id
            }
            if rgb == 0 {
                return 0; // empty subtree
            }
            node = rgb - 1; // internal: descend
        }
        0
    }

    /// Reconstruct a voxel's RGBA from a **streamed** `BrickVolume`: descend the
    /// octree to the brick id, then read the flat block array (brick `id` at
    /// `(id-1)·BRICK³·4`), exactly as the viewer addresses it.
    fn sample_streamed(bv: &BrickVolume, blocks: &[u8], x: u32, y: u32, z: u32) -> [u8; 4] {
        assert!(bv.streamed);
        let id = descend_octree(bv, x / BRICK, y / BRICK, z / BRICK);
        if id == 0 {
            return [0, 0, 0, 0];
        }
        let bk = BRICK;
        let local = (x % bk + (y % bk) * bk + (z % bk) * bk * bk) as usize;
        let off = ((id as usize - 1) * (bk as usize).pow(3) + local) * 4;
        [blocks[off], blocks[off + 1], blocks[off + 2], blocks[off + 3]]
    }

    #[test]
    fn streaming_builder_round_trips_in_hilbert_order() {
        // order_v=5 (32³); feed a contiguous Hilbert prefix (one byte/voxel),
        // spanning several bricks. The accumulator stays O(one brick).
        let order_v = 5;
        let n = 1u64 << (3 * 4); // 4096 voxels
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new());
        assert_eq!(b.open.len(), (BRICK as usize).pow(3), "accumulator is one brick");
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1); // byte > 0 ⇒ voxel occupied
        }
        let (bv, blocks) = b.finish_streaming().unwrap();
        assert!(bv.occupied >= 1);
        assert!(bv.streamed, "the --volume-res path ships the streamable format");
        // bricks.bin is exactly `occupied` flat brick blocks (streamed to `blocks`).
        assert!(bv.atlas.is_empty(), "streamed atlas lives on disk, not in the struct");
        assert_eq!(blocks.len(), bv.occupied as usize * (BRICK as usize).pow(3) * 4);
        // Every fed voxel reconstructs through the id page table + flat blocks.
        for h in 0..n {
            let v = hilbert_d2xyz(h, order_v);
            assert!(sample_streamed(&bv, &blocks, v[0], v[1], v[2])[3] > 0, "h={h} should be occupied");
        }
        // A voxel in an unfed (far) brick is empty.
        let far = hilbert_d2xyz((1u64 << (3 * order_v)) - 1, order_v);
        assert_eq!(sample_streamed(&bv, &blocks, far[0], far[1], far[2]), [0, 0, 0, 0]);
    }

    #[test]
    fn octree_indexes_every_occupied_brick_and_prunes_empty() {
        // order_v=6 (64³ voxels → 8³ = 512 brick cells, depth 3). Feed a
        // contiguous Hilbert prefix so several bricks fill; the octree must map
        // each occupied brick to a unique 1-based id and report empties as 0.
        let order_v = 6;
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new());
        let n = 1u64 << (3 * 5); // 32768 voxels
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1);
        }
        let (bv, _) = b.finish_streaming().unwrap();
        assert!(bv.streamed && bv.page_table.is_empty(), "streamed → octree, no flat page");
        assert_eq!(bv.tree_depth, order_v - BRICK.trailing_zeros());
        assert!(bv.node_count >= 1, "at least a root node");
        assert_eq!(bv.node_pool_dim[0] % 2, 0, "node pool is 2 texels/node/axis");

        // Every fed voxel's brick descends to a valid, in-range id.
        let mut seen = std::collections::HashSet::new();
        for h in 0..n {
            let v = hilbert_d2xyz(h, order_v);
            let id = descend_octree(&bv, v[0] / BRICK, v[1] / BRICK, v[2] / BRICK);
            assert!(id >= 1 && id <= bv.occupied, "h={h} → id {id} out of range");
            seen.insert((v[0] / BRICK, v[1] / BRICK, v[2] / BRICK));
        }
        // Distinct occupied brick cells == occupied count (a bijection cell↔id).
        assert_eq!(seen.len() as u32, bv.occupied);

        // A far, unfed brick cell prunes to empty (0).
        let far = hilbert_d2xyz((1u64 << (3 * order_v)) - 1, order_v);
        assert_eq!(descend_octree(&bv, far[0] / BRICK, far[1] / BRICK, far[2] / BRICK), 0);
    }

    #[test]
    fn octree_round_trips_at_high_depth() {
        // order_v=11 (2048³ voxels, depth 8) — the deep-tree case the viewer
        // must handle. Feed a contiguous prefix and confirm every occupied brick
        // still descends to a valid id and a far brick prunes to empty.
        let order_v = 11;
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new());
        let n = 1u64 << (3 * 6); // 262144 voxels → a small corner of the cube
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1);
        }
        let (bv, _) = b.finish_streaming().unwrap();
        assert_eq!(bv.tree_depth, 8);
        for h in 0..n {
            let v = hilbert_d2xyz(h, order_v);
            let id = descend_octree(&bv, v[0] / BRICK, v[1] / BRICK, v[2] / BRICK);
            assert!(id >= 1 && id <= bv.occupied, "h={h} → id {id} out of range at depth 8");
        }
        let far = hilbert_d2xyz((1u64 << (3 * order_v)) - 1, order_v);
        assert_eq!(descend_octree(&bv, far[0] / BRICK, far[1] / BRICK, far[2] / BRICK), 0);
    }

    #[test]
    fn streamed_rgba_bricks_round_trip_verbatim() {
        // Anisotropic extent forcing depth ≥ 2 (max page dim 5 → cube P=8, depth
        // 3). Scatter occupied voxels with distinct, non-round RGBA; every one
        // must round-trip EXACTLY through the octree + flat blocks — proving the
        // dense-RGBA builder stores color verbatim (unlike the byte path's
        // mean/activity/density transform).
        let extent = [24u32, 16, 40]; // page dims [3, 2, 5]
        let mut g = vec![0u8; (24 * 16 * 40 * 4) as usize];
        let put = |g: &mut [u8], x: u32, y: u32, z: u32, c: [u8; 4]| {
            let i = ((x + y * 24 + z * 24 * 16) * 4) as usize;
            g[i..i + 4].copy_from_slice(&c);
        };
        let pts = [
            ((1u32, 2u32, 3u32), [10u8, 20, 30, 200]),
            ((9, 5, 33), [1, 2, 3, 255]),
            ((23, 15, 39), [77, 88, 99, 100]), // far corner
            ((0, 0, 0), [255, 254, 253, 1]),
        ];
        for (p, c) in pts {
            put(&mut g, p.0, p.1, p.2, c);
        }
        let mut buf = Vec::new();
        let bv = build_streamed_brick_volume(&g, extent, BRICK, &mut buf).unwrap();
        assert!(bv.streamed);
        assert_eq!(bv.apron, 0, "streamed bricks carry no apron border");
        assert_eq!(bv.vol_dim, extent, "vol_dim is the full anisotropic extent");
        assert_eq!(bv.tree_depth, 3, "P = next_pow2(max([3,2,5])) = 8 → depth 3");
        assert_eq!(
            buf.len() as u32,
            bv.occupied * BRICK.pow(3) * 4,
            "bricks.bin is a flat occupied-block array"
        );
        for (p, c) in pts {
            assert_eq!(sample_streamed(&bv, &buf, p.0, p.1, p.2), c, "voxel {p:?} round-trips verbatim");
        }
        // An empty brick and an out-of-page-range cell both prune to empty.
        assert_eq!(sample_streamed(&bv, &buf, 16, 8, 8), [0, 0, 0, 0], "empty brick");
        assert_eq!(descend_octree(&bv, 7, 7, 7), 0, "cell beyond any page dim prunes");
    }

    #[test]
    fn streamed_rgba_octree_bijection_and_short_axis_pruning() {
        // Fill one voxel in each brick of a thin slab so occupied bricks span the
        // x axis but the short y/z axes have exactly one brick — their high cells
        // must prune to empty subtrees.
        let extent = [64u32, 8, 8]; // page dims [8, 1, 1] → cube P=8, depth 3
        let mut g = vec![0u8; (64 * 8 * 8 * 4) as usize];
        for bx in 0..8u32 {
            let i = (bx * BRICK * 4) as usize; // one voxel (x=bx·BRICK, y=z=0) per x-brick
            g[i..i + 4].copy_from_slice(&[1, 1, 1, 255]);
        }
        let mut buf = Vec::new();
        let bv = build_streamed_brick_volume(&g, extent, BRICK, &mut buf).unwrap();
        assert_eq!(bv.occupied, 8, "8 occupied bricks along x");
        // Distinct occupied cells ↔ ids 1..=occupied (a bijection).
        let mut ids = std::collections::HashSet::new();
        for bx in 0..8u32 {
            let id = descend_octree(&bv, bx, 0, 0);
            assert!(id >= 1 && id <= bv.occupied, "id {id} in range");
            ids.insert(id);
        }
        assert_eq!(ids.len() as u32, bv.occupied);
        // Short-axis high cells prune (only y=z=0 bricks exist).
        assert_eq!(descend_octree(&bv, 0, 1, 0), 0, "y beyond page dim prunes");
        assert_eq!(descend_octree(&bv, 0, 0, 1), 0, "z beyond page dim prunes");
    }

    #[test]
    fn streamed_rgba_empty_grid_has_no_bricks() {
        let extent = [40u32, 40, 40];
        let mut buf = Vec::new();
        let bv = build_streamed_brick_volume(&vec![0u8; 40 * 40 * 40 * 4], extent, BRICK, &mut buf).unwrap();
        assert_eq!(bv.occupied, 0);
        assert!(bv.streamed && bv.atlas.is_empty() && buf.is_empty());
    }

    #[test]
    fn stream_agg_slabs_match_single_pass() {
        // Same anisotropic grid as the verbatim round-trip test, with occupied
        // voxels on BOTH sides of a brick-aligned slab boundary (z=16). Bricking
        // it in two slabs must be byte-identical to a single full-extent pass —
        // proving id sequencing, octree order, and block offsets hold across slabs.
        let extent = [24u32, 16, 40];
        let (ex, ey) = (24usize, 16usize);
        let mut g = vec![0u8; ex * ey * 40 * 4];
        let put = |g: &mut [u8], x: u32, y: u32, z: u32, c: [u8; 4]| {
            let i = ((x + y * 24 + z * 24 * 16) * 4) as usize;
            g[i..i + 4].copy_from_slice(&c);
        };
        let pts = [
            ((1u32, 2u32, 3u32), [10u8, 20, 30, 200]), // slab [0,16)
            ((0, 0, 0), [255, 254, 253, 1]),           // slab [0,16)
            ((9, 5, 33), [1, 2, 3, 255]),              // slab [16,40)
            ((23, 15, 39), [77, 88, 99, 100]),         // slab [16,40)
        ];
        for (p, c) in pts {
            put(&mut g, p.0, p.1, p.2, c);
        }

        // Reference: one full-extent slab.
        let mut buf_full = Vec::new();
        let bv_full = build_streamed_brick_volume(&g, extent, BRICK, &mut buf_full).unwrap();

        // Two brick-aligned slabs, each fed only its own z-planes.
        let mut agg = StreamBrickAgg::new(extent, BRICK);
        let mut buf_slab = Vec::new();
        for (z0, z1) in [(0u32, 16u32), (16u32, 40u32)] {
            let depth = (z1 - z0) as usize;
            let mut slab = vec![0u8; ex * ey * depth * 4];
            for z in z0..z1 {
                let (src, dst) = ((z as usize) * ex * ey * 4, (z - z0) as usize * ex * ey * 4);
                slab[dst..dst + ex * ey * 4].copy_from_slice(&g[src..src + ex * ey * 4]);
            }
            agg.add_slab(&slab, z0, z1, &mut buf_slab).unwrap();
        }
        let bv_slab = agg.finish();

        assert_eq!(bv_slab.occupied, bv_full.occupied, "same occupied-brick count");
        assert_eq!(buf_slab, buf_full, "slab bricking is byte-identical to one pass");
        assert_eq!(bv_slab.node_pool, bv_full.node_pool, "octree node pool identical");
        assert_eq!(bv_slab.node_pool_dim, bv_full.node_pool_dim);
        assert_eq!(bv_slab.focus_center, bv_full.focus_center, "framing identical");
        assert_eq!(bv_slab.focus_radius, bv_full.focus_radius);
        for (p, c) in pts {
            assert_eq!(sample_streamed(&bv_slab, &buf_slab, p.0, p.1, p.2), c, "voxel {p:?}");
        }
    }

    #[test]
    fn octree_node_pool_is_sparse_not_dense() {
        // A single occupied brick in a large 256³-voxel volume (32³ = 32768 brick
        // cells) must build only a path of nodes (≈ depth), not a dense table.
        let order_v = 8; // 256³ voxels, depth 5, 32³ brick cells
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new());
        b.push(0, 1); // one occupied voxel at Hilbert distance 0 (origin brick)
        let (bv, _) = b.finish_streaming().unwrap();
        assert_eq!(bv.occupied, 1);
        // Depth-5 tree, one leaf → 5 nodes on the path (root + 4 internal).
        assert_eq!(bv.node_count, bv.tree_depth, "one leaf ⇒ one node per level");
        assert!(
            (bv.node_count as usize) < 32 * 32 * 32,
            "node pool is O(path), not O(brick cells)"
        );
        assert_eq!(descend_octree(&bv, 0, 0, 0), 1);
    }
}
