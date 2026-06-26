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
//! This v1 builds the pool by transforming a finished dense grid (so it is
//! bounded by the dense grid's resolution and never out-of-cores). Building the
//! pool directly from a *sparse* accumulator at higher virtual resolution — to
//! exceed the dense grid entirely — is the natural next step on this format.

/// Brick edge in voxels. `8³ = 512` voxels = 2 KiB RGBA8 per brick — a good
/// granularity for both empty-space skipping and the atlas.
pub const BRICK: u32 = 8;

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
    /// Number of occupied bricks (atlas slots used).
    pub occupied: u32,
}

impl BrickVolume {
    /// Atlas size in bricks per axis (`atlas_dim / BRICK`). The viewer derives
    /// this in JS; here it backs the format tests.
    #[cfg(test)]
    pub fn atlas_bricks(&self) -> [u32; 3] {
        [
            self.atlas_dim[0] / BRICK,
            self.atlas_dim[1] / BRICK,
            self.atlas_dim[2] / BRICK,
        ]
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

    // Near-cubic atlas big enough for every slot.
    let slots = occupied.max(1);
    let ax = (slots as f64).cbrt().ceil() as u32;
    let ax = ax.max(1);
    let ay = ax;
    let az = slots.div_ceil(ax * ay).max(1);
    let atlas_dim = [ax * brick, ay * brick, az * brick];
    let (adx, ady) = (atlas_dim[0] as usize, atlas_dim[1] as usize);
    let mut atlas = vec![0u8; (atlas_dim[0] as usize) * (atlas_dim[1] as usize) * (atlas_dim[2] as usize) * 4];

    // Pass 2: copy occupied voxels into their brick's atlas slot. Empty voxels
    // inside an occupied brick stay transparent (atlas is zero-initialized).
    for z in 0..ez {
        for y in 0..ey {
            for x in 0..ex {
                let src = (x as usize + y as usize * exs + z as usize * exs * eys) * 4;
                if rgba[src + 3] == 0 {
                    continue;
                }
                let slot = slot_of[page_idx(x / brick, y / brick, z / brick)] - 1; // 0-based
                let (sx, sy, sz) = (slot % ax, (slot / ax) % ay, slot / (ax * ay));
                let (axx, ayy, azz) = (
                    (sx * brick + x % brick) as usize,
                    (sy * brick + y % brick) as usize,
                    (sz * brick + z % brick) as usize,
                );
                let dst = (axx + ayy * adx + azz * adx * ady) * 4;
                atlas[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
            }
        }
    }

    // Page table: 1-based slot encoded little-endian in R,G,B (A = 255).
    let mut page_table = vec![0u8; npages * 4];
    for (i, &slot) in slot_of.iter().enumerate() {
        page_table[i * 4] = (slot & 0xff) as u8;
        page_table[i * 4 + 1] = ((slot >> 8) & 0xff) as u8;
        page_table[i * 4 + 2] = ((slot >> 16) & 0xff) as u8;
        page_table[i * 4 + 3] = 255;
    }

    BrickVolume {
        atlas,
        atlas_dim,
        page_table,
        page_dim: [pbx, pby, pbz],
        occupied,
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
        let (axx, ayy, azz) = (sx * BRICK + x % BRICK, sy * BRICK + y % BRICK, sz * BRICK + z % BRICK);
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
}
