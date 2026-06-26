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
//!   resolution (only the occupied-brick atlas itself grows with the data).

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

/// Cap on occupied bricks the streaming builder keeps (≈ a 512³-voxel atlas at
/// `BRICK=8`). Past this, [`BrickBuilder`] stops admitting new bricks and
/// counts them as dropped (logged) rather than ballooning the atlas/VRAM.
pub const MAX_BRICKS: u32 = 262_144;

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

/// The packed sparse volume: a brick-pool atlas + a page table indexing it.
pub struct BrickVolume {
    /// Brick-pool atlas, RGBA8, `atlas_dim` voxels (x-fastest). Occupied bricks
    /// are packed slot-by-slot; unused atlas space is left transparent.
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
    }
}

/// Streaming sparse-voxel brick builder for the byte→Hilbert pass. Points are
/// fed in non-decreasing Hilbert order at virtual order `order_v` (cube side
/// `2^order_v`); because an aligned `2^brick_log2` brick is a contiguous Hilbert
/// range, only the *current* brick is open at a time. Finished bricks stream
/// into a flat block list + a dense page table, assembled into a [`BrickVolume`]
/// by [`finish`](BrickBuilder::finish). Memory is `O(one brick)` plus the
/// page table and the occupied-brick output — never the full `2^order_v` cube.
pub struct BrickBuilder {
    order_v: u32,
    brick: u32,
    brick_log2: u32,
    luma: [u16; 256],
    page_dim: [u32; 3],
    page: Vec<u32>,    // dense (V/B)³: 0 = empty, else 1-based slot
    blocks: Vec<u8>,   // occupied bricks' RGBA8, B³ each, in slot order
    open: Vec<VoxelAcc>, // accumulators for the current open brick (B³)
    cur_hidx: i128,    // Hilbert index of the open brick (-1 = none)
    cur_origin: [u32; 3],
    occupied: u32,
    max_count: u64,
    dropped: u64,
}

impl BrickBuilder {
    /// `order_v` = virtual cube exponent (side `2^order_v`); `brick` = brick
    /// edge (power of two ≤ `2^order_v`); `luma` = per-byte luminance LUT.
    pub fn new(order_v: u32, brick: u32, luma: [u16; 256]) -> Self {
        assert!((1..=21).contains(&order_v));
        let brick_log2 = brick.trailing_zeros();
        let side = 1u32 << order_v;
        let pd = side / brick;
        BrickBuilder {
            order_v,
            brick,
            brick_log2,
            luma,
            page_dim: [pd, pd, pd],
            page: vec![0u32; (pd as usize).pow(3)],
            blocks: Vec::new(),
            open: vec![VoxelAcc::default(); (brick as usize).pow(3)],
            cur_hidx: -1,
            cur_origin: [0; 3],
            occupied: 0,
            max_count: 0,
            dropped: 0,
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

    /// Emit the open brick (if non-empty) into the block list + page table.
    fn finalize(&mut self) {
        if self.cur_hidx < 0 || self.open.iter().all(|a| a.count == 0) {
            return;
        }
        if self.occupied >= MAX_BRICKS {
            self.dropped += 1;
            return;
        }
        self.occupied += 1; // 1-based slot
        let bk = self.brick;
        let cell = [
            self.cur_origin[0] / bk,
            self.cur_origin[1] / bk,
            self.cur_origin[2] / bk,
        ];
        let pidx = cell[0] as usize
            + cell[1] as usize * self.page_dim[0] as usize
            + cell[2] as usize * self.page_dim[0] as usize * self.page_dim[1] as usize;
        self.page[pidx] = self.occupied;
        for acc in &self.open {
            if acc.count == 0 {
                self.blocks.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                let c = acc.count as u64;
                let mean = (acc.sum_val / c).min(255) as u8;
                let act = (acc.sum_luma / c).min(255) as u8;
                self.blocks
                    .extend_from_slice(&[mean, act, c.min(255) as u8, 255]);
                self.max_count = self.max_count.max(c);
            }
        }
    }

    /// Finish: assemble the flat blocks into a 3D atlas (rescaling the density
    /// channel by the global max count) and the dense page table.
    pub fn finish(mut self) -> (BrickVolume, u64) {
        self.finalize();
        let bk = self.brick;
        let [axb, ayb, azb] = atlas_dims_bricks(self.occupied);
        let atlas_dim = [axb * bk, ayb * bk, azb * bk];
        let (adx, ady) = (atlas_dim[0] as usize, atlas_dim[1] as usize);
        let mut atlas = vec![0u8; (atlas_dim[0] * atlas_dim[1] * atlas_dim[2]) as usize * 4];
        let inv_max = if self.max_count > 0 {
            255.0 / self.max_count as f32
        } else {
            0.0
        };
        let bsz = (bk as usize).pow(3) * 4; // bytes per flat brick block
        for slot in 0..self.occupied {
            let (sx, sy, sz) = (slot % axb, (slot / axb) % ayb, slot / (axb * ayb));
            let block = &self.blocks[slot as usize * bsz..(slot as usize + 1) * bsz];
            for lz in 0..bk {
                for ly in 0..bk {
                    for lx in 0..bk {
                        let si = ((lx + ly * bk + lz * bk * bk) * 4) as usize;
                        if block[si + 3] == 0 {
                            continue;
                        }
                        let (axx, ayy, azz) =
                            ((sx * bk + lx) as usize, (sy * bk + ly) as usize, (sz * bk + lz) as usize);
                        let di = (axx + ayy * adx + azz * adx * ady) * 4;
                        atlas[di] = block[si];
                        atlas[di + 1] = block[si + 1];
                        // Density (B): rescale the raw count by the global max.
                        atlas[di + 2] = (block[si + 2] as f32 * inv_max).round().clamp(0.0, 255.0) as u8;
                        atlas[di + 3] = 255;
                    }
                }
            }
        }
        let side = 1u32 << self.order_v;
        let bv = BrickVolume {
            atlas,
            atlas_dim,
            page_table: pack_page_table(&self.page),
            page_dim: self.page_dim,
            vol_dim: [side, side, side],
            apron: 0, // streaming can't see neighbors → no border, nearest filtering
            occupied: self.occupied,
        };
        (bv, self.dropped)
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

    #[test]
    fn streaming_builder_round_trips_in_hilbert_order() {
        // order_v=5 (32³); feed a contiguous Hilbert prefix (one byte/voxel),
        // spanning several bricks. The accumulator stays O(one brick).
        let order_v = 5;
        let n = 1u64 << (3 * 4); // 4096 voxels
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256]);
        assert_eq!(b.open.len(), (BRICK as usize).pow(3), "accumulator is one brick");
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1); // byte > 0 ⇒ voxel occupied
        }
        let (bv, dropped) = b.finish();
        assert_eq!(dropped, 0);
        assert!(bv.occupied >= 1);
        let ext = [1u32 << order_v; 3];
        // Every fed voxel reconstructs through page table + atlas.
        for h in 0..n {
            let v = hilbert_d2xyz(h, order_v);
            assert!(sample(&bv, ext, v[0], v[1], v[2])[3] > 0, "h={h} should be occupied");
        }
        // A voxel in an unfed (far) brick is empty.
        let far = hilbert_d2xyz((1u64 << (3 * order_v)) - 1, order_v);
        assert_eq!(sample(&bv, ext, far[0], far[1], far[2]), [0, 0, 0, 0]);
    }
}
