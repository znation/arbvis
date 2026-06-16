use fast_hilbert::h2xy;
use std::collections::HashMap;

// Hilbert sub-quadrant state table.
// CHILD_TABLE[state][i] = (dx, dy, child_state) where (dx,dy) ∈ {0,1}²
// give the child quadrant's position within the parent (units of child_side).
// Derived from fast_hilbert's order-1 LUT.
pub const CHILD_TABLE: [[(u32, u32, u8); 4]; 4] = [
    [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 2)], // state 0
    [(0, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 3)], // state 1
    [(1, 1, 3), (0, 1, 2), (0, 0, 2), (1, 0, 0)], // state 2
    [(1, 1, 2), (1, 0, 3), (0, 0, 3), (0, 1, 1)], // state 3
];

/// Recursively decompose Hilbert local range [a, b) at `level` (order-`level`
/// curve on a 2^level × 2^level square) into axis-aligned pixel rectangles.
pub fn decompose_hilbert(
    a: u64,
    b: u64,
    level: u8,
    x0: u32,
    y0: u32,
    side: u32,
    state: u8,
    out: &mut Vec<(u32, u32, u32, u32)>,
) {
    let total = (side as u64) * (side as u64);
    if a == 0 && b == total {
        out.push((x0, y0, x0 + side, y0 + side));
        return;
    }
    if level == 0 {
        out.push((x0, y0, x0 + 1, y0 + 1));
        return;
    }
    let child_side = side >> 1;
    let q_size = (child_side as u64) * (child_side as u64);
    for i in 0u64..4 {
        let ca = i * q_size;
        let cb = ca + q_size;
        if a >= cb || b <= ca {
            continue;
        }
        let (dx, dy, child_state) = CHILD_TABLE[state as usize][i as usize];
        decompose_hilbert(
            a.saturating_sub(ca),
            b.min(cb) - ca,
            level - 1,
            x0 + dx * child_side,
            y0 + dy * child_side,
            child_side,
            child_state,
            out,
        );
    }
}

/// Compute the XOR-merged set of a list of intervals: ranges covered by an odd
/// number of input intervals. Used for extracting outer boundary edges.
pub fn xor_intervals(intervals: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut events: Vec<(u32, i8)> = Vec::with_capacity(intervals.len() * 2);
    for &(lo, hi) in intervals {
        if lo < hi {
            events.push((lo, 1));
            events.push((hi, -1));
        }
    }
    events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut result = Vec::new();
    let mut count: i32 = 0;
    let mut seg_start = 0u32;
    for (y, delta) in &events {
        let was_odd = count % 2 != 0;
        count += *delta as i32;
        let is_odd = count % 2 != 0;
        if !was_odd && is_odd {
            seg_start = *y;
        } else if was_odd && !is_odd && seg_start < *y {
            result.push((seg_start, *y));
        }
    }
    result
}

/// Compute outer boundary segments of a set of axis-aligned pixel rectangles
/// that exactly tile a region. Returns (x0,y0,x1,y1) segments in pixel-boundary
/// coords: a rect [px, px+w) × [py, py+h) has edges at x=px, x=px+w, y=py, y=py+h.
pub fn outer_segments(rects: &[(u32, u32, u32, u32)]) -> Vec<(u32, u32, u32, u32)> {
    let mut vert: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut horiz: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for &(x0, y0, x1, y1) in rects {
        vert.entry(x0).or_default().push((y0, y1));
        vert.entry(x1).or_default().push((y0, y1));
        horiz.entry(y0).or_default().push((x0, x1));
        horiz.entry(y1).or_default().push((x0, x1));
    }
    let mut result = Vec::new();
    for (&x, intervals) in &vert {
        for (lo, hi) in xor_intervals(intervals) {
            result.push((x, lo, x, hi));
        }
    }
    for (&y, intervals) in &horiz {
        for (lo, hi) in xor_intervals(intervals) {
            result.push((lo, y, hi, y));
        }
    }
    result
}

/// Collect all dyadic pixel rectangles for a file's byte range.
pub fn file_rects(
    byte_start: u64,
    byte_end: u64,
    total_pixels: u64,
    square_pixels: u64,
    num_squares: u32,
    height: u32,
    kh: u8,
) -> Vec<(u32, u32, u32, u32)> {
    let byte_end = byte_end.min(total_pixels);
    if byte_end <= byte_start {
        return Vec::new();
    }
    let mut all_rects = Vec::new();
    for sq in 0..num_squares as u64 {
        let sq_start = sq * square_pixels;
        let sq_end = sq_start + square_pixels;
        let local_a = byte_start.max(sq_start).saturating_sub(sq_start);
        let local_b = byte_end.min(sq_end).saturating_sub(sq_start);
        if local_b <= local_a {
            continue;
        }
        let x_off = sq as u32 * height;
        decompose_hilbert(local_a, local_b, kh, x_off, 0, height, 0, &mut all_rects);
    }
    all_rects
}

/// Area-weighted centroid of a set of axis-aligned pixel rectangles.
pub fn rects_centroid(rects: &[(u32, u32, u32, u32)]) -> Option<(u32, u32)> {
    let mut total_area = 0f64;
    let mut wx = 0f64;
    let mut wy = 0f64;
    for &(x0, y0, x1, y1) in rects {
        let area = (x1 - x0) as f64 * (y1 - y0) as f64;
        wx += area * (x0 + x1) as f64 / 2.0;
        wy += area * (y0 + y1) as f64 / 2.0;
        total_area += area;
    }
    if total_area == 0.0 {
        return None;
    }
    Some(((wx / total_area) as u32, (wy / total_area) as u32))
}

/// Determine a consistent hue for a file based on its name.
pub fn name_hue(name: &str) -> u16 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    (h.finish() % 360) as u16
}

/// Hilbert curve index → pixel coordinates using u64 for large curvers.
pub fn hilbert_to_xy_u64(idx: u64, order: u8) -> (u32, u32) {
    h2xy::<u32>(idx, order)
}

/// Number of axes for the 3D Hilbert curve.
const N3: usize = 3;

/// Smallest curve order `bits` such that a `2^bits` cube holds at least
/// `cells` points (i.e. `2^(3*bits) >= cells`). Used to pick the 3D Hilbert
/// order for a voxel grid of a given side. Clamped to `[1, 21]` — order 21
/// fills a `2^21` cube and uses 63 of a `u64`'s 64 bits.
pub fn hilbert3d_order_for_cells(cells: u64) -> u32 {
    let mut bits = 1u32;
    while (bits < 21) && ((1u128 << (3 * bits)) < cells as u128) {
        bits += 1;
    }
    bits
}

/// 3D Hilbert curve: map a 1D distance `h` to cube coordinates `(x, y, z)`,
/// each in `[0, 2^bits)`. Inverse of [`hilbert_xyz2d`].
///
/// Hand-rolled Skilling transform (J. Skilling, "Programming the Hilbert
/// curve", 2004): de-interleave `h` into a per-axis "transpose", then run the
/// recursion-free TransposeToAxes pass. `fast_hilbert` is 2D-only and the
/// n-dimensional crates pull in `BigUint`; this stays on plain integers, is
/// allocation-free, and is exercised by the round-trip tests below.
pub fn hilbert_d2xyz(h: u64, bits: u32) -> [u32; N3] {
    debug_assert!((1..=21).contains(&bits));
    // De-interleave the distance into the transpose X[0..3]. Convention (matches
    // the reference `hilbertcurve` port): bit `b` of axis `i` is bit
    // `b*3 + (2 - i)` of `h`, i.e. axis 0 carries the most significant of each
    // interleaved triple.
    let mut x = [0u32; N3];
    for b in 0..bits {
        for (i, xi) in x.iter_mut().enumerate() {
            let src = (b as usize) * N3 + (N3 - 1 - i);
            *xi |= (((h >> src) & 1) as u32) << b;
        }
    }
    // TransposeToAxes.
    let top: u32 = 1 << bits; // sentinel = 2^bits
    let t = x[N3 - 1] >> 1; // inverse gray code
    for i in (1..N3).rev() {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;
    let mut q: u32 = 2;
    while q != top {
        let p = q - 1;
        for i in (0..N3).rev() {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let s = (x[0] ^ x[i]) & p;
                x[0] ^= s;
                x[i] ^= s;
            }
        }
        q <<= 1;
    }
    x
}

/// 3D Hilbert curve: map cube coordinates `(x, y, z)`, each in `[0, 2^bits)`,
/// to the 1D distance along the curve. Inverse of [`hilbert_d2xyz`].
///
/// The forward map ([`hilbert_d2xyz`]) is all the render path needs; this
/// inverse completes the API and anchors the round-trip property tests.
#[allow(dead_code)]
pub fn hilbert_xyz2d(mut x: [u32; N3], bits: u32) -> u64 {
    debug_assert!((1..=21).contains(&bits));
    // AxesToTranspose.
    let mut q: u32 = 1 << (bits - 1); // top bit
    while q > 1 {
        let p = q - 1;
        for i in 0..N3 {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    for i in 1..N3 {
        x[i] ^= x[i - 1]; // gray code
    }
    // Skilling's final fixup: fold the top axis's high bits back across all
    // axes (the inverse of TransposeToAxes's leading gray-decode).
    let mut t = 0u32;
    let mut qq: u32 = 1 << (bits - 1);
    while qq > 1 {
        if x[N3 - 1] & qq != 0 {
            t ^= qq - 1;
        }
        qq >>= 1;
    }
    for xi in x.iter_mut() {
        *xi ^= t;
    }
    // Interleave the transpose back into the distance (inverse of the
    // de-interleave in `hilbert_d2xyz`).
    let mut h = 0u64;
    for b in 0..bits {
        for (i, &xi) in x.iter().enumerate() {
            let dst = (b as usize) * N3 + (N3 - 1 - i);
            h |= (((xi >> b) & 1) as u64) << dst;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_intervals_no_overlap() {
        let intervals = vec![(0, 5), (6, 10)];
        let result = xor_intervals(&intervals);
        assert_eq!(result, vec![(0, 5), (6, 10)]);
    }

    #[test]
    fn test_xor_intervals_overlap_cancels() {
        let intervals = vec![(0, 5), (3, 5)]; // (0,3) appears once, (3,5) appears twice → cancels
        let result = xor_intervals(&intervals);
        assert_eq!(result, vec![(0, 3)]);
    }

    #[test]
    fn test_xor_intervals_empty() {
        assert!(xor_intervals(&[]).is_empty());
    }

    #[test]
    fn test_decompose_hilbert_full_square() {
        let mut rects = Vec::new();
        decompose_hilbert(0, 4, 1, 0, 0, 2, 0, &mut rects);
        assert_eq!(rects, vec![(0, 0, 2, 2)]);
    }

    #[test]
    fn test_rects_centroid_single_rect() {
        let rects = vec![(10, 10, 20, 20)];
        let c = rects_centroid(&rects);
        assert_eq!(c, Some((15, 15)));
    }

    #[test]
    fn test_rects_centroid_empty() {
        assert_eq!(rects_centroid(&[]), None);
    }

    #[test]
    fn test_rects_centroid_zero_area() {
        assert_eq!(rects_centroid(&[(0, 0, 0, 0)]), None);
    }

    #[test]
    fn test_name_hue_consistent() {
        let h1 = name_hue("foo");
        let h2 = name_hue("foo");
        assert_eq!(h1, h2);
        assert!(h1 < 360);
    }

    #[test]
    fn test_hilbert_to_xy_u64_bounds() {
        let (x, y) = hilbert_to_xy_u64(0, 8);
        assert_eq!((x, y), (0, 0));
        let side = 1u32 << 8;
        let (x2, y2) = hilbert_to_xy_u64((side * side - 1) as u64, 8);
        assert!(x2 < side);
        assert!(y2 < side);
    }

    #[test]
    fn test_hilbert3d_order_for_cells() {
        assert_eq!(hilbert3d_order_for_cells(0), 1);
        assert_eq!(hilbert3d_order_for_cells(8), 1); // 2^3 == 8
        assert_eq!(hilbert3d_order_for_cells(9), 2); // needs 4^3 = 64
        assert_eq!(hilbert3d_order_for_cells(64), 2);
        assert_eq!(hilbert3d_order_for_cells(65), 3);
        assert_eq!(hilbert3d_order_for_cells(1 << 24), 8); // 256^3
        assert_eq!(hilbert3d_order_for_cells(u64::MAX), 21); // clamp
    }

    #[test]
    fn test_hilbert3d_origin() {
        for bits in 1..=8 {
            assert_eq!(hilbert_d2xyz(0, bits), [0, 0, 0]);
            assert_eq!(hilbert_xyz2d([0, 0, 0], bits), 0);
        }
    }

    #[test]
    fn test_hilbert3d_roundtrip_exhaustive_small() {
        // Exhaustively round-trip every distance for small orders.
        for bits in 1..=5 {
            let cells = 1u64 << (3 * bits);
            for h in 0..cells {
                let p = hilbert_d2xyz(h, bits);
                let side = 1u32 << bits;
                assert!(
                    p[0] < side && p[1] < side && p[2] < side,
                    "coord OOB at h={h} bits={bits}: {p:?}"
                );
                assert_eq!(
                    hilbert_xyz2d(p, bits),
                    h,
                    "roundtrip failed h={h} bits={bits}"
                );
            }
        }
    }

    #[test]
    fn test_hilbert3d_bijection_small() {
        // Every distance maps to a distinct coordinate (bijection onto the cube).
        for bits in 1..=5 {
            let cells = 1u64 << (3 * bits);
            let side = 1u64 << bits;
            let mut seen = std::collections::HashSet::with_capacity(cells as usize);
            for h in 0..cells {
                let p = hilbert_d2xyz(h, bits);
                let key = (p[0] as u64) * side * side + (p[1] as u64) * side + p[2] as u64;
                assert!(seen.insert(key), "duplicate coord at h={h} bits={bits}");
            }
            assert_eq!(seen.len() as u64, cells);
        }
    }

    #[test]
    fn test_hilbert3d_adjacency() {
        // Hilbert locality: consecutive distances are unit-distance neighbors.
        for bits in 1..=6 {
            let cells = 1u64 << (3 * bits);
            for h in 1..cells {
                let a = hilbert_d2xyz(h - 1, bits);
                let b = hilbert_d2xyz(h, bits);
                let d = (a[0].abs_diff(b[0])) + (a[1].abs_diff(b[1])) + (a[2].abs_diff(b[2]));
                assert_eq!(d, 1, "non-adjacent step h={h} bits={bits}: {a:?} -> {b:?}");
            }
        }
    }

    #[test]
    fn test_hilbert3d_roundtrip_large_orders() {
        // Sample larger orders (can't enumerate); round-trip a deterministic
        // pseudo-random spread of distances. Uses a tiny xorshift so there's no
        // rng dependency.
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for bits in [10u32, 15, 21] {
            let max = if 3 * bits >= 64 {
                u64::MAX
            } else {
                (1u64 << (3 * bits)) - 1
            };
            for _ in 0..5000 {
                let h = next() & max;
                let p = hilbert_d2xyz(h, bits);
                let side = 1u64 << bits;
                assert!((p[0] as u64) < side && (p[1] as u64) < side && (p[2] as u64) < side);
                assert_eq!(
                    hilbert_xyz2d(p, bits),
                    h,
                    "roundtrip failed h={h} bits={bits}"
                );
            }
        }
    }
}
