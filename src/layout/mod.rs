//! Mapping from canvas pixel coordinates to tensor element coordinates.
//!
//! Two variants:
//!
//! - [`Layout::HilbertGlobal`] preserves the legacy byte-Hilbert behaviour:
//!   one continuous space-filling curve over the concatenated byte stream of
//!   every source. 1 px = 1 byte. Used for non-safetensors inputs and when
//!   the user passes `--layout hilbert`.
//!
//! - [`Layout::Architectural`] is the structure-aware mode: each tensor
//!   occupies a rectangle of its natural 2D element shape (1 px = 1 element);
//!   transformer blocks are stacked vertically and pixel-aligned across the
//!   stack. Used when every source is safetensors and tensor names look
//!   transformer-style, unless overridden.

pub mod arch;
pub mod bin_pack;
pub mod hilbert;
pub mod model_config;
pub mod name_tree;
pub mod render;

use crate::data::{Source, SourceMeta};
use crate::safetensors::Dtype;

/// One contiguous (row-major) element range of one tensor that overlaps a
/// tile. The tile renderer fetches `byte_start..byte_end` from the source
/// and decodes elements at the natural dtype stride.
#[derive(Debug, Clone)]
pub struct TileRegion {
    /// Index into the source list this region's bytes come from.
    pub source_idx: usize,
    /// Index into the global tensor list (across all sources, in canvas order).
    pub tensor_id: usize,
    pub dtype: Dtype,
    /// Tensor row × col element-grid dimensions, as `TensorMeta::element_shape`.
    pub tensor_rows: u64,
    pub tensor_cols: u64,
    /// First/last row of the tensor element grid that overlap this tile.
    pub row_first: u64,
    pub row_last_exclusive: u64,
    /// First/last column of the tensor element grid that overlap this tile.
    pub col_first: u64,
    pub col_last_exclusive: u64,
    /// Byte offset within the source where this tensor's element data starts.
    pub tensor_byte_start: u64,
    /// Canvas pixel rectangle this region paints into, in *tile-local* (px, py)
    /// coordinates `[x0, x1) × [y0, y1)`.
    pub tile_x0: u32,
    pub tile_y0: u32,
    pub tile_x1: u32,
    pub tile_y1: u32,
}

/// User-facing layout selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    /// Auto-select: architectural if every source is safetensors AND
    /// transformer-style names are detected; otherwise hilbert.
    Auto,
    /// Force architectural (structure-aware) layout. Falls back to hilbert if
    /// no source is safetensors.
    Arch,
    /// Force the legacy global-Hilbert layout.
    Hilbert,
}

impl Default for LayoutMode {
    fn default() -> Self { LayoutMode::Auto }
}

/// The chosen layout for the current run.
pub enum Layout {
    HilbertGlobal(hilbert::HilbertLayout),
    Architectural(arch::ArchLayout),
}

impl Layout {
    /// Total canvas dimensions, in pixels.
    pub fn canvas_size(&self) -> (u32, u32) {
        match self {
            Layout::HilbertGlobal(h) => h.canvas_size(),
            Layout::Architectural(a) => a.canvas_size(),
        }
    }

    /// Total pixel count (= width * height).
    pub fn total_pixels(&self) -> u64 {
        let (w, h) = self.canvas_size();
        w as u64 * h as u64
    }

    /// Whether this layout's per-pixel rendering needs source byte access
    /// (Plain / Diff / Xet) or can be drawn from layout metadata alone (Dtype).
    pub fn needs_bytes_for_plain(&self) -> bool { true }

    /// Whether this is the architectural variant (drives a few mode branches
    /// downstream — e.g. element-per-pixel decoding vs byte-per-pixel LUT).
    pub fn is_architectural(&self) -> bool {
        matches!(self, Layout::Architectural(_))
    }
}

/// Build the layout for the given sources. Returns `Layout::HilbertGlobal`
/// for non-architectural runs.
///
/// `metas` is an optional list parallel to `sources` carrying opportunistic
/// `config.json` / `model.safetensors.index.json` data. When non-empty it
/// lets the architectural layout: (1) validate the inferred layer count and
/// extend the canonical layer stack to `num_hidden_layers` when only a
/// subset of shards was loaded; (2) extend the canonical sub-path set with
/// tensor names listed in the index but not loaded. Pass `&[]` to opt out.
pub fn select_layout(
    sources: &[Source],
    cumulative_offsets: &[u64],
    total_bytes: u64,
    mode: LayoutMode,
    metas: &[SourceMeta],
) -> Layout {
    // Force-hilbert path: use legacy byte-Hilbert geometry as-is.
    if matches!(mode, LayoutMode::Hilbert) {
        return Layout::HilbertGlobal(hilbert::HilbertLayout::from_total(total_bytes));
    }

    // Architectural requires every source to be safetensors AND a detectable
    // structure. If forced (`Arch`) but the data doesn't support it, we still
    // fall back to hilbert with a log line — accidentally rendering nothing is
    // a worse outcome than ignoring the forced flag.
    let all_safetensors = !sources.is_empty()
        && sources.iter().all(|s| s.safetensors.is_some());

    if all_safetensors {
        if let Some(arch) = arch::ArchLayout::try_build(sources, cumulative_offsets, metas) {
            return Layout::Architectural(arch);
        }
        if matches!(mode, LayoutMode::Arch) {
            log::warn!("--layout arch requested but no recognisable structure; falling back to hilbert");
        }
    } else if matches!(mode, LayoutMode::Arch) {
        log::warn!("--layout arch requested but not every input is safetensors; falling back to hilbert");
    }

    Layout::HilbertGlobal(hilbert::HilbertLayout::from_total(total_bytes))
}
