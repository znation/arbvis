use fast_hilbert::h2xy;

fn main() {
    for k in [8u32, 9, 10, 11, 12] {
        let side = 1u32 << k;
        let canvas = (side * side) as usize;
        println!("k={}, side={}, canvas={}", k, side, canvas);
        let max_idx = canvas - 1;
        let (x, y) = h2xy::<u32>(max_idx as u64, 1);
        println!("  order=1 max_idx={} -> x={} y={} (in bounds: {})", max_idx, x, y, x < side && y < side);
        let (x2, y2) = h2xy::<u32>(max_idx as u64, k as u8);
        println!("  order=k max_idx={} -> x={} y={} (in bounds: {})", max_idx, x2, y2, x2 < side && y2 < side);
    }
}
