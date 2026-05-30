//! Simple 2D shelf packer used to lay out tensor slots inside a layer block.
//!
//! Not optimal — alignment matters more than density for this viz, since the
//! canonical layer arrangement should repeat identically across all of
//! `model.layers.0..N-1`. We pack slots row-by-row left-to-right, opening a
//! new shelf when the current row overflows `max_width`.

#[derive(Debug, Clone, Copy)]
pub struct Slot {
    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Pack `slots` (preserving input order) into a canvas at most `max_width` wide.
/// Returns a `Placement` per slot and the bounding `(width, height)` of the
/// packed region. `padding` pixels separate adjacent slots and shelves.
pub fn pack(slots: &[Slot], max_width: u32, padding: u32) -> (Vec<Placement>, u32, u32) {
    let mut out: Vec<Placement> = Vec::with_capacity(slots.len());
    let mut x_cursor: u32 = 0;
    let mut y_cursor: u32 = 0;
    let mut shelf_height: u32 = 0;
    let mut max_x_seen: u32 = 0;

    for s in slots {
        // Force a wrap if the next slot wouldn't fit on the current shelf.
        if x_cursor > 0 && x_cursor + s.width > max_width {
            y_cursor += shelf_height + padding;
            x_cursor = 0;
            shelf_height = 0;
        }
        out.push(Placement {
            x: x_cursor,
            y: y_cursor,
            width: s.width,
            height: s.height,
        });
        x_cursor += s.width + padding;
        if s.height > shelf_height {
            shelf_height = s.height;
        }
        if x_cursor > max_x_seen {
            max_x_seen = x_cursor;
        }
    }

    let total_height = y_cursor + shelf_height;
    // Trim the trailing padding we added past the last slot.
    let total_width = max_x_seen.saturating_sub(padding);
    (out, total_width, total_height)
}

/// Round `n` up to the next multiple of `align`. `align` must be a power of two.
#[inline]
pub fn align_up(n: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn pack_single_slot_no_padding() {
        let slots = vec![Slot {
            width: 100,
            height: 50,
        }];
        let (placements, w, h) = pack(&slots, 1000, 0);
        assert_eq!(placements.len(), 1);
        assert_eq!((placements[0].x, placements[0].y), (0, 0));
        assert_eq!((w, h), (100, 50));
    }

    #[test]
    fn pack_two_slots_fit_one_shelf() {
        let slots = vec![
            Slot {
                width: 100,
                height: 50,
            },
            Slot {
                width: 100,
                height: 60,
            },
        ];
        let (placements, w, h) = pack(&slots, 500, 4);
        assert_eq!(placements[0].x, 0);
        assert_eq!(placements[1].x, 104);
        assert_eq!(h, 60);
        assert!((204..=208).contains(&w));
    }

    #[test]
    fn pack_wraps_when_overflow() {
        let slots = vec![
            Slot {
                width: 100,
                height: 50,
            },
            Slot {
                width: 100,
                height: 60,
            },
            Slot {
                width: 100,
                height: 40,
            },
        ];
        let (placements, _, h) = pack(&slots, 250, 4);
        // First two fit on shelf 0; third wraps to shelf 1 because 100+4+100+4+100 > 250.
        assert_eq!(placements[0].y, 0);
        assert_eq!(placements[1].y, 0);
        assert_eq!(placements[2].y, 60 + 4);
        assert_eq!(h, 60 + 4 + 40);
    }

    #[test]
    fn pack_preserves_input_order() {
        let slots = vec![
            Slot {
                width: 10,
                height: 10,
            },
            Slot {
                width: 20,
                height: 10,
            },
            Slot {
                width: 30,
                height: 10,
            },
            Slot {
                width: 40,
                height: 10,
            },
        ];
        let (placements, _, _) = pack(&slots, 1000, 0);
        // x positions are monotonically increasing for a single-shelf pack.
        let xs: Vec<u32> = placements.iter().map(|p| p.x).collect();
        assert_eq!(xs, vec![0, 10, 30, 60]);
    }

    #[test]
    fn pack_empty() {
        let (placements, w, h) = pack(&[], 100, 4);
        assert!(placements.is_empty());
        assert_eq!((w, h), (0, 0));
    }
}
