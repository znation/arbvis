//! Streaming brick **writers** — the builders that emit bricks to a [`Write`]
//! sink (`bricks.bin`) as they finalize instead of accumulating the atlas in
//! RAM: [`StreamBrickAgg`] for the dense/structured (`rgb`) path and
//! [`BrickBuilder`] for the byte→Hilbert pass. The shared on-disk format
//! ([`BrickVolume`], the sparse octree) lives in [`super::brick`].

use std::io::Write;

use super::brick::{BrickVolume, Octree};
use super::encode::VoxelAcc;
use crate::geometry::{hilbert3d_node_origin, hilbert_d2xyz};

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

/// Fold one occupied brick at voxel-space `origin` (side `brick`) into the
/// occupied-region framing stats shared by [`StreamBrickAgg`] and
/// [`BrickBuilder`]: a running bbox min/max plus a centroid sum that weights
/// each brick at its voxel-space center (`origin + brick/2`).
fn track_occupied_brick(
    origin: [u32; 3],
    brick: u32,
    fmin: &mut [u32; 3],
    fmax: &mut [u32; 3],
    fsum: &mut [f64; 3],
) {
    for a in 0..3 {
        fmin[a] = fmin[a].min(origin[a]);
        fmax[a] = fmax[a].max(origin[a] + brick - 1);
        fsum[a] += (origin[a] + brick / 2) as f64;
    }
}

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
        debug_assert!(
            z0.is_multiple_of(self.brick),
            "slab boundary must be brick-aligned"
        );
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
                    track_occupied_brick(
                        [bx * brick, by * brick, bz * brick],
                        brick,
                        &mut self.fmin,
                        &mut self.fmax,
                        &mut self.fsum,
                    );
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
            atlas_dim: [0, 0, 0],   // streamed: bricks.bin is a flat block array
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
    cur_hidx: i128,     // Hilbert index of the open brick (-1 = none)
    cur_origin: [u32; 3],
    occupied: u32,
    max_count: u64,
    // Voxel-space bounding box + centroid of occupied bricks, for framing the
    // camera on the *fine* data (see BrickVolume::focus_center).
    fmin: [u32; 3],
    fmax: [u32; 3],
    fsum: [f64; 3],
    // Bulk mode (`bulk` = true): several bytes map to the same voxel on average
    // (`total > cells_v` at the call site), so per byte we look the local voxel
    // index up in `lut` — rebuilt once per brick transition — instead of paying
    // a full `hilbert_d2xyz` decode per byte. `lut_hidx` is the brick the LUT
    // was built for.
    bulk: bool,
    lut: Vec<u32>,
    lut_hidx: i128,
}

impl<W: Write> BrickBuilder<W> {
    /// `order_v` = virtual cube exponent (side `2^order_v`); `brick` = brick
    /// edge (power of two ≤ `2^order_v`); `luma` = per-byte luminance LUT;
    /// `writer` = sink for the flat brick blocks (`bricks.bin`). `bulk` enables
    /// the per-brick voxel-index LUT (see the struct field docs); set it when
    /// several bytes share a voxel on average (`total > cells_v`).
    pub fn new(order_v: u32, brick: u32, luma: [u16; 256], writer: W, bulk: bool) -> Self {
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
            bulk,
            lut: Vec::new(),
            lut_hidx: -1,
        }
    }

    /// Build the bulk-mode LUT: for the brick `hidx`, map each of the `brick³`
    /// in-brick Hilbert offsets to the brick-local voxel index. Correct because
    /// the local index depends on `h` only through (brick, low bits) — the brick
    /// origin is fixed for the whole brick — so one decode per in-brick offset
    /// serves every byte that lands there.
    fn build_lut(&mut self, hidx: i128) {
        let n = 1usize << (3 * self.brick_log2);
        let base = (hidx as u64) << (3 * self.brick_log2);
        let bk = self.brick;
        let o = self.cur_origin;
        self.lut.clear();
        for low in 0..n {
            let v = hilbert_d2xyz(base | low as u64, self.order_v);
            self.lut
                .push((v[0] - o[0]) + (v[1] - o[1]) * bk + (v[2] - o[2]) * bk * bk);
        }
        self.lut_hidx = hidx;
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
            if self.bulk {
                self.build_lut(hidx);
            }
        }
        let li = if self.bulk {
            debug_assert_eq!(self.lut_hidx, hidx);
            self.lut[(h & ((1u64 << (3 * self.brick_log2)) - 1)) as usize] as usize
        } else {
            let v = hilbert_d2xyz(h, self.order_v);
            let bk = self.brick;
            ((v[0] - self.cur_origin[0])
                + (v[1] - self.cur_origin[1]) * bk
                + (v[2] - self.cur_origin[2]) * bk * bk) as usize
        };
        let acc = &mut self.open[li];
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
        track_occupied_brick(
            self.cur_origin,
            bk,
            &mut self.fmin,
            &mut self.fmax,
            &mut self.fsum,
        );
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
            atlas: Vec::new(),      // bricks were streamed to `writer` (bricks.bin)
            atlas_dim: [0, 0, 0],   // streamed: bricks.bin is a flat block array
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
    use crate::geometry::hilbert_d2xyz;
    use crate::volume::brick::BRICK;

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
        [
            blocks[off],
            blocks[off + 1],
            blocks[off + 2],
            blocks[off + 3],
        ]
    }

    #[test]
    fn bulk_lut_matches_per_byte_mode() {
        // Bulk mode must be byte-for-byte equivalent to the per-byte decode:
        // feed the same non-decreasing Hilbert sequence with repeated voxels
        // (the `total > cells_v` regime) through both and compare the streamed
        // bricks.bin output.
        let order_v = 7;
        let n = 1u64 << (3 * 6); // 64³ voxel range, spanning many 8³ bricks
        let mut per_byte = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new(), false);
        let mut bulk = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new(), true);
        for h in 0..n {
            let byte = ((h & 0x7f) as u8) | 1;
            per_byte.push(h, byte);
            bulk.push(h, byte);
            if h % 3 == 0 {
                per_byte.push(h, byte.wrapping_add(1)); // repeat voxel: bpv > 1
                bulk.push(h, byte.wrapping_add(1));
            }
        }
        let (_, a) = per_byte.finish_streaming().unwrap();
        let (_, b) = bulk.finish_streaming().unwrap();
        assert_eq!(a, b, "bulk LUT path must match the per-byte decode");
    }

    #[test]
    fn streaming_builder_round_trips_in_hilbert_order() {
        // order_v=5 (32³); feed a contiguous Hilbert prefix (one byte/voxel),
        // spanning several bricks. The accumulator stays O(one brick).
        let order_v = 5;
        let n = 1u64 << (3 * 4); // 4096 voxels
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new(), false);
        assert_eq!(
            b.open.len(),
            (BRICK as usize).pow(3),
            "accumulator is one brick"
        );
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1); // byte > 0 ⇒ voxel occupied
        }
        let (bv, blocks) = b.finish_streaming().unwrap();
        assert!(bv.occupied >= 1);
        assert!(
            bv.streamed,
            "the --volume-res path ships the streamable format"
        );
        // bricks.bin is exactly `occupied` flat brick blocks (streamed to `blocks`).
        assert!(
            bv.atlas.is_empty(),
            "streamed atlas lives on disk, not in the struct"
        );
        assert_eq!(
            blocks.len(),
            bv.occupied as usize * (BRICK as usize).pow(3) * 4
        );
        // Every fed voxel reconstructs through the id page table + flat blocks.
        for h in 0..n {
            let v = hilbert_d2xyz(h, order_v);
            assert!(
                sample_streamed(&bv, &blocks, v[0], v[1], v[2])[3] > 0,
                "h={h} should be occupied"
            );
        }
        // A voxel in an unfed (far) brick is empty.
        let far = hilbert_d2xyz((1u64 << (3 * order_v)) - 1, order_v);
        assert_eq!(
            sample_streamed(&bv, &blocks, far[0], far[1], far[2]),
            [0, 0, 0, 0]
        );
    }

    #[test]
    fn octree_indexes_every_occupied_brick_and_prunes_empty() {
        // order_v=6 (64³ voxels → 8³ = 512 brick cells, depth 3). Feed a
        // contiguous Hilbert prefix so several bricks fill; the octree must map
        // each occupied brick to a unique 1-based id and report empties as 0.
        let order_v = 6;
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new(), false);
        let n = 1u64 << (3 * 5); // 32768 voxels
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1);
        }
        let (bv, _) = b.finish_streaming().unwrap();
        assert!(
            bv.streamed && bv.page_table.is_empty(),
            "streamed → octree, no flat page"
        );
        assert_eq!(bv.tree_depth, order_v - BRICK.trailing_zeros());
        assert!(bv.node_count >= 1, "at least a root node");
        assert_eq!(
            bv.node_pool_dim[0] % 2,
            0,
            "node pool is 2 texels/node/axis"
        );

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
        assert_eq!(
            descend_octree(&bv, far[0] / BRICK, far[1] / BRICK, far[2] / BRICK),
            0
        );
    }

    #[test]
    fn octree_round_trips_at_high_depth() {
        // order_v=11 (2048³ voxels, depth 8) — the deep-tree case the viewer
        // must handle. Feed a contiguous prefix and confirm every occupied brick
        // still descends to a valid id and a far brick prunes to empty.
        let order_v = 11;
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new(), false);
        let n = 1u64 << (3 * 6); // 262144 voxels → a small corner of the cube
        for h in 0..n {
            b.push(h, ((h & 0x7f) as u8) | 1);
        }
        let (bv, _) = b.finish_streaming().unwrap();
        assert_eq!(bv.tree_depth, 8);
        for h in 0..n {
            let v = hilbert_d2xyz(h, order_v);
            let id = descend_octree(&bv, v[0] / BRICK, v[1] / BRICK, v[2] / BRICK);
            assert!(
                id >= 1 && id <= bv.occupied,
                "h={h} → id {id} out of range at depth 8"
            );
        }
        let far = hilbert_d2xyz((1u64 << (3 * order_v)) - 1, order_v);
        assert_eq!(
            descend_octree(&bv, far[0] / BRICK, far[1] / BRICK, far[2] / BRICK),
            0
        );
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
        assert_eq!(
            bv.tree_depth, 3,
            "P = next_pow2(max([3,2,5])) = 8 → depth 3"
        );
        assert_eq!(
            buf.len() as u32,
            bv.occupied * BRICK.pow(3) * 4,
            "bricks.bin is a flat occupied-block array"
        );
        for (p, c) in pts {
            assert_eq!(
                sample_streamed(&bv, &buf, p.0, p.1, p.2),
                c,
                "voxel {p:?} round-trips verbatim"
            );
        }
        // An empty brick and an out-of-page-range cell both prune to empty.
        assert_eq!(
            sample_streamed(&bv, &buf, 16, 8, 8),
            [0, 0, 0, 0],
            "empty brick"
        );
        assert_eq!(
            descend_octree(&bv, 7, 7, 7),
            0,
            "cell beyond any page dim prunes"
        );
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
        let bv = build_streamed_brick_volume(&vec![0u8; 40 * 40 * 40 * 4], extent, BRICK, &mut buf)
            .unwrap();
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

        assert_eq!(
            bv_slab.occupied, bv_full.occupied,
            "same occupied-brick count"
        );
        assert_eq!(
            buf_slab, buf_full,
            "slab bricking is byte-identical to one pass"
        );
        assert_eq!(
            bv_slab.node_pool, bv_full.node_pool,
            "octree node pool identical"
        );
        assert_eq!(bv_slab.node_pool_dim, bv_full.node_pool_dim);
        assert_eq!(
            bv_slab.focus_center, bv_full.focus_center,
            "framing identical"
        );
        assert_eq!(bv_slab.focus_radius, bv_full.focus_radius);
        for (p, c) in pts {
            assert_eq!(
                sample_streamed(&bv_slab, &buf_slab, p.0, p.1, p.2),
                c,
                "voxel {p:?}"
            );
        }
    }

    #[test]
    fn octree_node_pool_is_sparse_not_dense() {
        // A single occupied brick in a large 256³-voxel volume (32³ = 32768 brick
        // cells) must build only a path of nodes (≈ depth), not a dense table.
        let order_v = 8; // 256³ voxels, depth 5, 32³ brick cells
        let mut b = BrickBuilder::new(order_v, BRICK, [3u16; 256], Vec::new(), false);
        b.push(0, 1); // one occupied voxel at Hilbert distance 0 (origin brick)
        let (bv, _) = b.finish_streaming().unwrap();
        assert_eq!(bv.occupied, 1);
        // Depth-5 tree, one leaf → 5 nodes on the path (root + 4 internal).
        assert_eq!(
            bv.node_count, bv.tree_depth,
            "one leaf ⇒ one node per level"
        );
        assert!(
            (bv.node_count as usize) < 32 * 32 * 32,
            "node pool is O(path), not O(brick cells)"
        );
        assert_eq!(descend_octree(&bv, 0, 0, 0), 1);
    }
}
