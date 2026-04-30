use std::collections::HashMap;
use fast_hilbert::h2xy;

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

/// Count multiples of `stride` in the byte range `[byte_start, byte_end)`.
pub fn sampled_in_range(byte_start: u64, byte_end: u64, stride: u64) -> u64 {
    if byte_end <= byte_start {
        return 0;
    }
    if stride == 1 {
        return byte_end - byte_start;
    }
    if byte_start == 0 {
        (byte_end - 1) / stride + 1
    } else {
        (byte_end - 1) / stride - (byte_start - 1) / stride
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampled_in_range_basic() {
        assert_eq!(sampled_in_range(0, 0, 1), 0);
        assert_eq!(sampled_in_range(0, 10, 1), 10);
        assert_eq!(sampled_in_range(0, 10, 3), 4); // 0,3,6,9
        assert_eq!(sampled_in_range(1, 10, 3), 3); // 3,6,9
        assert_eq!(sampled_in_range(5, 10, 2), 2); // 6,8
    }

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
}