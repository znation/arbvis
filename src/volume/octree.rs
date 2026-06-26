//! Streaming Hilbert **point-LOD octree** builder — the 3D analog of the 2D
//! tile pyramid, and the foundation of the streamed high-resolution 3D view.
//!
//! # Why this exists
//!
//! The legacy point cloud ([`super::mod`]) stride-subsamples the whole stream to
//! a flat 1.5M-point budget and ships it wholesale: zooming in never reveals
//! more detail. This module instead organizes points into a multiresolution
//! **octree** whose nodes are streamed on demand, so the viewer fetches only the
//! nodes in view at the zoom-appropriate resolution and converges to one point
//! per byte when zoomed in — exactly how the 2D Leaflet pyramid behaves.
//!
//! # The free octree
//!
//! Because arbvis lays bytes along a 3D Hilbert curve, a contiguous index range
//! whose length is a power of eight is *already* an axis-aligned octree
//! sub-cube (see [`crate::geometry::hilbert3d_node_origin`]). So the octree is
//! the Hilbert ordering itself: no sort, no second pass. A node at `depth` with
//! `node_idx` owns Hilbert range `[node_idx·8^(P-depth), (node_idx+1)·8^(P-depth))`
//! on a `2^P` cube (`P` = [`PointOctreeBuilder::order`]). Children are the eight
//! sub-ranges; `child_idx = node_idx*8 + octant`.
//!
//! # LOD by accept-or-descend (Potree-style)
//!
//! Points arrive in non-decreasing Hilbert order. Each node owns a coarse
//! occupancy grid of `G³` cells (`G = 2^grid_log2`, capped at the node side); a
//! point is *accepted* at the shallowest ancestor whose cell is still free, else
//! it *descends*. Coarse levels are thus spatially-uniform subsamples and leaves
//! hold the residue down to one point per voxel. A node holds at most `G³`
//! points, so children are created lazily only when a node overflows.
//!
//! Memory is bounded by the open root→leaf stack (`≤ P+1` nodes), independent of
//! input size — the whole point of streaming.
//!
//! # On disk
//!
//! [`PointOctree::data`] concatenates every node's point block (positions
//! quantized to per-node-local `u8`/`u16` coords + RGBA8); [`PointOctree::records`]
//! is one fixed-size [`NodeRecord`] per node. The viewer locates a node's bytes
//! by `byte_offset`/`byte_length` and reconstructs its cube from
//! `(node_idx, depth, order)` — no explicit child pointers, no ordering
//! requirement.

use crate::geometry::{hilbert3d_node_origin, hilbert_d2xyz, hilbert_xyz2d};

/// Default occupancy-grid exponent: `G = 2^5 = 32`, so a node caps at `32³ =
/// 32768` points. Small enough to keep per-node download snappy, large enough
/// that interior nodes stay sparse.
pub const POINT_GRID_LOG2: u32 = 5;

/// Serialized size of one [`NodeRecord`] (little-endian, fixed-width).
pub const RECORD_SIZE: usize = 40;

/// One octree node's metadata. Written verbatim to the `*_hierarchy.bin` index.
///
/// `origin` + `depth` (+ the bundle's `order`) fully describe the node's cube,
/// so the viewer rebuilds the tree spatially — a node's parent is the depth-1
/// node whose cube contains it — needing no Hilbert decode or 64-bit arithmetic
/// in JS. `node_idx`/`child_mask` are kept as the canonical id and a
/// has-children hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeRecord {
    /// Hilbert octree node index (base-8 path digits, most significant first).
    pub node_idx: u64,
    /// Offset of this node's point block within [`PointOctree::data`].
    pub byte_offset: u64,
    /// Block length in bytes (`point_count · stride`).
    pub byte_length: u32,
    pub point_count: u32,
    /// Min-corner voxel of the node's cube (on the `2^order` grid). Side is
    /// `2^(order-depth)`.
    pub origin: [u32; 3],
    pub depth: u8,
    /// Bit `k` set ⇒ child octant `k` (node `node_idx*8 + k`) exists.
    pub child_mask: u8,
    /// Per-axis local-coordinate width: `8` or `16` bits.
    pub coord_bits: u8,
}

impl NodeRecord {
    /// Bytes per point in this node's block: 3 local coords + RGBA8. (The
    /// shipping viewer parses blocks in JS; this Rust mirror is exercised by
    /// the format tests.)
    #[cfg(test)]
    pub fn stride(&self) -> usize {
        3 * (self.coord_bits as usize / 8) + 4
    }

    fn write_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.node_idx.to_le_bytes());
        out.extend_from_slice(&self.byte_offset.to_le_bytes());
        out.extend_from_slice(&self.byte_length.to_le_bytes());
        out.extend_from_slice(&self.point_count.to_le_bytes());
        out.extend_from_slice(&self.origin[0].to_le_bytes());
        out.extend_from_slice(&self.origin[1].to_le_bytes());
        out.extend_from_slice(&self.origin[2].to_le_bytes());
        out.push(self.depth);
        out.push(self.child_mask);
        out.push(self.coord_bits);
        out.push(0); // pad to RECORD_SIZE
    }

    /// Parse one record from a `RECORD_SIZE`-byte little-endian slice — the
    /// inverse of [`write_le`](Self::write_le), mirroring what the JS viewer
    /// does. Used by the format round-trip test.
    #[cfg(test)]
    pub fn read_le(b: &[u8]) -> Self {
        let u64le = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let u32le = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        NodeRecord {
            node_idx: u64le(0),
            byte_offset: u64le(8),
            byte_length: u32le(16),
            point_count: u32le(20),
            origin: [u32le(24), u32le(28), u32le(32)],
            depth: b[36],
            child_mask: b[37],
            coord_bits: b[38],
        }
    }
}

/// The built point octree: a flat data buffer + the node index.
pub struct PointOctree {
    /// Concatenated per-node point blocks.
    pub data: Vec<u8>,
    /// One record per node (emission order; no ordering guarantee).
    pub records: Vec<NodeRecord>,
    /// Curve order `P`: the virtual point grid is `2^P` per axis.
    pub order: u32,
    /// Points dropped as exact-voxel duplicates (kept for logging).
    pub dropped: u64,
}

impl PointOctree {
    /// Serialize [`Self::records`] into the `*_hierarchy.bin` payload.
    pub fn serialize_hierarchy(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.records.len() * RECORD_SIZE);
        for r in &self.records {
            r.write_le(&mut out);
        }
        out
    }

    /// Total stored (post-LOD) point count across all nodes.
    pub fn total_points(&self) -> u64 {
        self.records.iter().map(|r| r.point_count as u64).sum()
    }
}

/// An open node on the root→current-leaf stack during the build.
struct OpenNode {
    node_idx: u64,
    depth: u32,
    /// Min-corner voxel of the node's cube.
    origin: [u32; 3],
    /// Node cube side in voxels (`2^(order-depth)`).
    side: u32,
    /// Right-shift mapping a local voxel coord to its occupancy-cell coord.
    cell_shift: u32,
    /// Occupancy grid edge in cells (`min(G, side)`).
    g_eff: u32,
    /// One bit per occupancy cell (`g_eff³` bits).
    occ: Vec<u64>,
    /// Accepted points: `(local voxel coord, rgba)`.
    pts: Vec<([u32; 3], [u8; 4])>,
    child_mask: u8,
}

impl OpenNode {
    fn new(node_idx: u64, depth: u32, order: u32, grid_log2: u32) -> Self {
        let side_log2 = order - depth;
        let side = 1u32 << side_log2;
        // Occupancy grid edge: G capped at the node side (a node smaller than
        // the grid stores one point per voxel — exact).
        let ge_log2 = grid_log2.min(side_log2);
        let g_eff = 1u32 << ge_log2;
        let cells = (g_eff as usize).pow(3);
        OpenNode {
            node_idx,
            depth,
            origin: hilbert3d_node_origin(node_idx, depth, order),
            side,
            cell_shift: side_log2 - ge_log2,
            g_eff,
            occ: vec![0u64; cells.div_ceil(64)],
            pts: Vec::new(),
            child_mask: 0,
        }
    }

    /// Occupancy-cell index of a local voxel coordinate.
    fn cell_index(&self, local: [u32; 3]) -> usize {
        let cx = (local[0] >> self.cell_shift) as usize;
        let cy = (local[1] >> self.cell_shift) as usize;
        let cz = (local[2] >> self.cell_shift) as usize;
        let g = self.g_eff as usize;
        (cz * g + cy) * g + cx
    }

    fn occ_get(&self, cell: usize) -> bool {
        (self.occ[cell >> 6] >> (cell & 63)) & 1 != 0
    }

    fn occ_set(&mut self, cell: usize) {
        self.occ[cell >> 6] |= 1u64 << (cell & 63);
    }
}

/// Streaming builder. Push points in non-decreasing Hilbert order, then
/// [`finish`](PointOctreeBuilder::finish).
pub struct PointOctreeBuilder {
    order: u32,
    grid_log2: u32,
    open: Vec<OpenNode>,
    data: Vec<u8>,
    records: Vec<NodeRecord>,
    dropped: u64,
    last_h: Option<u64>,
}

impl PointOctreeBuilder {
    /// `order` = Hilbert order `P` (virtual grid `2^P` per axis, `1..=21`);
    /// `grid_log2` = occupancy-grid exponent (use [`POINT_GRID_LOG2`]).
    pub fn new(order: u32, grid_log2: u32) -> Self {
        assert!((1..=21).contains(&order), "point order out of range: {order}");
        PointOctreeBuilder {
            order,
            grid_log2,
            open: Vec::with_capacity(order as usize + 1),
            data: Vec::new(),
            records: Vec::new(),
            dropped: 0,
            last_h: None,
        }
    }

    /// Accept one point at Hilbert distance `h` (on the `2^order` grid) with
    /// color `rgba`. `h` must be non-decreasing across calls.
    pub fn push(&mut self, h: u64, rgba: [u8; 4]) {
        let order = self.order;
        // The Hilbert map is a bijection, so equal distances are the same voxel,
        // and (non-decreasing input) they arrive consecutively — dedup them to
        // one representative per voxel (first color wins), like voxel aggregation.
        if self.last_h == Some(h) {
            self.dropped += 1;
            return;
        }
        self.last_h = Some(h);
        let v = hilbert_d2xyz(h, order);

        // 1. Pop the divergent suffix: the longest common prefix of the open
        //    stack with `h`'s root→leaf path stays; deeper nodes have closed.
        let mut lcp = 0;
        while lcp < self.open.len() {
            let want = h >> (3 * (order - self.open[lcp].depth));
            if self.open[lcp].node_idx == want {
                lcp += 1;
            } else {
                break;
            }
        }
        while self.open.len() > lcp {
            let node = self.open.pop().unwrap();
            if let Some(parent) = self.open.last_mut() {
                parent.child_mask |= 1u8 << (node.node_idx & 7);
            }
            self.finalize(node);
        }

        // 2. Accept-or-descend from the root, materializing path nodes lazily.
        let mut di = 0usize;
        loop {
            if di == self.open.len() {
                let depth = di as u32;
                let node_idx = h >> (3 * (order - depth));
                self.open
                    .push(OpenNode::new(node_idx, depth, order, self.grid_log2));
            }
            let node = &mut self.open[di];
            let local = [
                v[0] - node.origin[0],
                v[1] - node.origin[1],
                v[2] - node.origin[2],
            ];
            let cell = node.cell_index(local);
            if !node.occ_get(cell) {
                node.occ_set(cell);
                node.pts.push((local, rgba));
                return;
            }
            if node.depth == order {
                // Leaf voxel already taken ⇒ an exact-position duplicate.
                self.dropped += 1;
                return;
            }
            di += 1;
        }
    }

    /// Flush the remaining open stack (deepest-first, so children finalize
    /// before parents and `child_mask`s are complete) and return the octree.
    pub fn finish(mut self) -> PointOctree {
        while let Some(node) = self.open.pop() {
            if let Some(parent) = self.open.last_mut() {
                parent.child_mask |= 1u8 << (node.node_idx & 7);
            }
            self.finalize(node);
        }
        PointOctree {
            data: self.data,
            records: self.records,
            order: self.order,
            dropped: self.dropped,
        }
    }

    /// Quantize a node's points into its on-disk block and emit its record.
    fn finalize(&mut self, node: OpenNode) {
        if node.pts.is_empty() {
            return; // never created an empty node, but be defensive
        }
        // Local coords are in `[0, side)`. Pick the narrowest width that holds
        // them; quantize only the rare shallow node whose side exceeds u16.
        let coord_bits: u8 = if node.side <= 256 { 8 } else { 16 };
        let byte_offset = self.data.len() as u64;
        match coord_bits {
            8 => {
                for (local, rgba) in &node.pts {
                    self.data.push(local[0] as u8);
                    self.data.push(local[1] as u8);
                    self.data.push(local[2] as u8);
                    self.data.extend_from_slice(rgba);
                }
            }
            _ => {
                let quant = |c: u32| -> u16 {
                    if node.side <= 65536 {
                        c as u16 // exact
                    } else {
                        // Coarse shallow node: scale [0,side) → [0,65535].
                        ((c as u64 * 65535) / (node.side as u64 - 1)) as u16
                    }
                };
                for (local, rgba) in &node.pts {
                    self.data
                        .extend_from_slice(&quant(local[0]).to_le_bytes());
                    self.data
                        .extend_from_slice(&quant(local[1]).to_le_bytes());
                    self.data
                        .extend_from_slice(&quant(local[2]).to_le_bytes());
                    self.data.extend_from_slice(rgba);
                }
            }
        }
        let point_count = node.pts.len() as u32;
        let stride = 3 * (coord_bits as u32 / 8) + 4;
        self.records.push(NodeRecord {
            node_idx: node.node_idx,
            byte_offset,
            byte_length: point_count * stride,
            point_count,
            origin: node.origin,
            depth: node.depth as u8,
            child_mask: node.child_mask,
            coord_bits,
        });
    }
}

/// Build a point octree from points that are **not** already Hilbert-ordered —
/// the general spatial-octree build (the byte floor skips this; its stream is
/// born Hilbert-ordered). Positions are normalized to `[0,1]` per axis of the
/// (possibly anisotropic) box; they're quantized onto a `2^order` cube, sorted
/// by Hilbert distance, then streamed through [`PointOctreeBuilder`]. Used by
/// the structured/entity path so a layout's per-element points get the same
/// LOD streaming as the byte cloud.
pub fn build_from_normalized_points(
    points: &[([f32; 3], [u8; 4])],
    order: u32,
    grid_log2: u32,
) -> PointOctree {
    let span = (1u64 << order) as f32;
    let maxc = (1u32 << order) - 1;
    let q = |f: f32| ((f.clamp(0.0, 1.0) * span) as u32).min(maxc);
    let mut items: Vec<(u64, [u8; 4])> = points
        .iter()
        .map(|(p, c)| (hilbert_xyz2d([q(p[0]), q(p[1]), q(p[2])], order), *c))
        .collect();
    items.sort_unstable_by_key(|x| x.0);
    let mut b = PointOctreeBuilder::new(order, grid_log2);
    for (h, c) in items {
        b.push(h, c);
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{hilbert3d_node_origin, hilbert_d2xyz};
    use std::collections::{HashMap, HashSet};

    /// Reconstruct the absolute voxel coordinates of a node's stored points by
    /// slicing its byte range out of the data buffer — exactly what a
    /// range-request viewer does.
    fn node_voxels(oct: &PointOctree, r: &NodeRecord) -> Vec<[u32; 3]> {
        // The stored origin must match the Hilbert recompute — the viewer trusts
        // the stored value, so cross-check it here.
        let origin = r.origin;
        assert_eq!(origin, hilbert3d_node_origin(r.node_idx, r.depth as u32, oct.order));
        let side = 1u32 << (oct.order - r.depth as u32);
        let stride = r.stride();
        let block = &oct.data[r.byte_offset as usize..(r.byte_offset + r.byte_length as u64) as usize];
        assert_eq!(block.len(), r.point_count as usize * stride, "block size matches count");
        let mut out = Vec::with_capacity(r.point_count as usize);
        for p in block.chunks_exact(stride) {
            let local = if r.coord_bits == 8 {
                [p[0] as u32, p[1] as u32, p[2] as u32]
            } else {
                let rd = |o: usize| u16::from_le_bytes([p[o], p[o + 1]]) as u32;
                if side <= 65536 {
                    [rd(0), rd(2), rd(4)]
                } else {
                    let deq = |q: u32| (q as u64 * (side as u64 - 1) / 65535) as u32;
                    [deq(rd(0)), deq(rd(2)), deq(rd(4))]
                }
            };
            out.push([origin[0] + local[0], origin[1] + local[1], origin[2] + local[2]]);
        }
        out
    }

    /// Build an octree from every Hilbert distance in `0..count` (each a
    /// distinct voxel), with a deterministic color.
    fn build_dense(order: u32, grid_log2: u32, count: u64) -> PointOctree {
        let mut b = PointOctreeBuilder::new(order, grid_log2);
        for h in 0..count {
            let c = (h & 0xff) as u8;
            b.push(h, [c, c ^ 0x55, c ^ 0xaa, 255]);
        }
        b.finish()
    }

    #[test]
    fn accounts_for_every_distinct_point_exactly_once() {
        // order=6 (64³), small grid (G=4) so the tree is several levels deep and
        // genuinely subsamples at coarse nodes.
        let order = 6;
        let count = 1u64 << (3 * 4); // 4096 distinct points (a Hilbert prefix)
        let oct = build_dense(order, 2, count);
        assert_eq!(oct.dropped, 0, "distinct h ⇒ no duplicates dropped");

        // Union of all nodes' reconstructed voxels == the input voxel set, once.
        let mut seen: HashSet<[u32; 3]> = HashSet::new();
        let mut total = 0u64;
        for r in &oct.records {
            for v in node_voxels(&oct, r) {
                assert!(seen.insert(v), "voxel {v:?} stored in more than one node");
                total += 1;
            }
        }
        assert_eq!(total, count, "every point stored exactly once");
        let expected: HashSet<[u32; 3]> = (0..count).map(|h| hilbert_d2xyz(h, order)).collect();
        assert_eq!(seen, expected, "stored voxels == input voxels");
    }

    #[test]
    fn duplicates_at_a_voxel_are_dropped() {
        let mut b = PointOctreeBuilder::new(5, 2);
        // Push the same distance three times plus two others.
        for h in [0u64, 0, 0, 7, 7] {
            b.push(h, [1, 2, 3, 255]);
        }
        let oct = b.finish();
        assert_eq!(oct.dropped, 3, "two extra 0s + one extra 7");
        assert_eq!(oct.total_points(), 2, "two distinct voxels survive");
    }

    #[test]
    fn range_fetch_returns_points_inside_node_cube() {
        let order = 7;
        let oct = build_dense(order, 3, 1u64 << (3 * 5)); // 32768 points
        for r in &oct.records {
            let origin = hilbert3d_node_origin(r.node_idx, r.depth as u32, order);
            let side = 1u32 << (order - r.depth as u32);
            for v in node_voxels(&oct, r) {
                for a in 0..3 {
                    assert!(
                        v[a] >= origin[a] && v[a] < origin[a] + side,
                        "point {v:?} escaped node cube origin {origin:?} side {side}"
                    );
                }
            }
        }
    }

    #[test]
    fn node_point_count_is_bounded_by_grid() {
        let grid_log2 = 3;
        let cap = (1u32 << grid_log2).pow(3); // G³ = 512
        let oct = build_dense(7, grid_log2, 1u64 << (3 * 5));
        for r in &oct.records {
            assert!(
                r.point_count <= cap,
                "node depth {} holds {} > cap {cap}",
                r.depth,
                r.point_count
            );
        }
    }

    #[test]
    fn coarse_root_subsamples_when_data_exceeds_capacity() {
        let grid_log2 = 2; // G=4 ⇒ root cap 64
        let count = 1u64 << (3 * 4); // 4096 points ≫ 64
        let oct = build_dense(6, grid_log2, count);
        let root = oct
            .records
            .iter()
            .find(|r| r.depth == 0)
            .expect("root node exists");
        assert!(root.point_count <= 64, "root capped at G³=64");
        assert!(
            oct.records.len() > 1,
            "overflow forces a multi-node tree, got {}",
            oct.records.len()
        );
    }

    #[test]
    fn child_mask_is_consistent_with_the_node_set() {
        let order = 6;
        let oct = build_dense(order, 2, 1u64 << (3 * 4));
        let present: HashMap<(u8, u64), NodeRecord> =
            oct.records.iter().map(|r| ((r.depth, r.node_idx), *r)).collect();

        for r in &oct.records {
            // Every child_mask bit points at a node that exists.
            for k in 0..8u64 {
                if r.child_mask & (1 << k) != 0 {
                    let key = (r.depth + 1, r.node_idx * 8 + k);
                    assert!(present.contains_key(&key), "missing child {key:?} of {r:?}");
                }
            }
            // Every non-root node's parent exists and claims it.
            if r.depth > 0 {
                let parent = present
                    .get(&(r.depth - 1, r.node_idx / 8))
                    .unwrap_or_else(|| panic!("missing parent of {r:?}"));
                let octant = (r.node_idx & 7) as u8;
                assert!(
                    parent.child_mask & (1 << octant) != 0,
                    "parent {parent:?} doesn't claim child octant {octant}"
                );
            }
        }
    }

    #[test]
    fn hierarchy_round_trips_through_bytes() {
        let oct = build_dense(5, 2, 200);
        let bytes = oct.serialize_hierarchy();
        assert_eq!(bytes.len(), oct.records.len() * RECORD_SIZE);
        for (i, r) in oct.records.iter().enumerate() {
            let parsed = NodeRecord::read_le(&bytes[i * RECORD_SIZE..(i + 1) * RECORD_SIZE]);
            assert_eq!(parsed, *r);
        }
    }

    #[test]
    fn build_from_normalized_points_is_a_valid_octree() {
        // Pseudo-random points in [0,1]^3 (not Hilbert-ordered).
        let mut s: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut pts = Vec::new();
        for _ in 0..5000 {
            let f = |n: u64| (n % 100_000) as f32 / 100_000.0;
            pts.push(([f(next()), f(next()), f(next())], [(next() % 256) as u8, 9, 9, 255]));
        }
        let order = 7;
        let oct = build_from_normalized_points(&pts, order, 3);
        assert_eq!(oct.order, order);
        assert!(oct.total_points() > 0 && oct.total_points() <= pts.len() as u64);
        assert!(oct.records.iter().any(|r| r.depth == 0), "has a root");
        // Same structural invariants as the streaming build: points lie in
        // their node's cube (node_voxels cross-checks origin via Hilbert too).
        for r in &oct.records {
            let side = 1u32 << (order - r.depth as u32);
            for v in node_voxels(&oct, r) {
                for a in 0..3 {
                    assert!(v[a] >= r.origin[a] && v[a] < r.origin[a] + side);
                }
            }
        }
    }

    #[test]
    fn small_data_collapses_to_a_single_exact_root() {
        // order ≤ grid_log2 ⇒ root grid is at least as fine as the cube, so a
        // small prefix stays entirely at the root with exact coords.
        let oct = build_dense(4, 5, 1u64 << (3 * 3)); // 512 points, grid finer than 16³
        assert_eq!(oct.records.len(), 1, "everything fits the root");
        assert_eq!(oct.records[0].depth, 0);
        assert_eq!(oct.records[0].coord_bits, 8, "side 16 ≤ 256 ⇒ 8-bit coords");
    }
}
