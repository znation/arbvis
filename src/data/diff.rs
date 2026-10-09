//! Crosshatch fill color for `UnmatchedRegion` / `OneSidedRange` sources —
//! the diff path uses these to mark one-side-only spans visually. Split out
//! of `data/mod.rs` so the source/IO half and the diff half each read in one
//! sitting.
//!
//! The diff-source builders (`JsonDiffBuilder`, `PlainBytesDiffBuilder`) and
//! the directory byte-diff walker live in `crate::data_diff`; `data/mod.rs`
//! re-exports them so `crate::data::…` paths stay valid.

/// Crosshatch fill color for `UnmatchedRegion` / `OneSidedRange` sources —
/// the diff path uses these to mark one-side-only spans visually.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiffFill {
    Grey,
    Red,
    Green,
}

impl DiffFill {
    /// Fill for original-side-only regions: grey when `is_finetune` marks an
    /// expected finetune drop, red otherwise. The diff paths in
    /// `crate::data_diff` and `crate::json_diff` share this mapping.
    pub fn orig_only(is_finetune: bool) -> DiffFill {
        if is_finetune {
            DiffFill::Grey
        } else {
            DiffFill::Red
        }
    }

    /// `(stripe, base)` colors for the crosshatch pattern. `stripe` is the
    /// foreground diagonal line color; `base` is the fill behind it.
    pub fn colors(self) -> (image::Rgb<u8>, image::Rgb<u8>) {
        match self {
            DiffFill::Grey => (image::Rgb([80, 80, 80]), image::Rgb([160, 160, 160])),
            DiffFill::Red => (image::Rgb([120, 0, 0]), image::Rgb([220, 40, 40])),
            DiffFill::Green => (image::Rgb([0, 120, 0]), image::Rgb([40, 220, 40])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DiffFill;

    #[test]
    fn diff_fill_colors_distinguish_sides() {
        let (grey_stripe, _) = DiffFill::Grey.colors();
        let (red_stripe, _) = DiffFill::Red.colors();
        let (green_stripe, _) = DiffFill::Green.colors();
        assert_ne!(grey_stripe, red_stripe);
        assert_ne!(red_stripe, green_stripe);
        assert_ne!(grey_stripe, green_stripe);
    }

    #[test]
    fn orig_only_fill_maps_finetune_to_grey() {
        assert_eq!(DiffFill::orig_only(true), DiffFill::Grey);
        assert_eq!(DiffFill::orig_only(false), DiffFill::Red);
    }
}
