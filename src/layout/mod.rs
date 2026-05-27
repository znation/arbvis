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
use crate::format::Dtype;

/// One contiguous (row-major) element range of one tensor that overlaps a
/// tile. The tile renderer fetches `byte_start..byte_end` from the source
/// and decodes elements at the natural dtype stride.
///
/// The tensor is drawn at a per-tensor display footprint that may differ from
/// its element grid (see [`crate::layout::arch::PlacedTensor`]): shrunk for
/// huge tensors, enlarged for thin vectors. At a given zoom level the footprint
/// is additionally multiplied by `2^(zoom - max_zoom)`. The renderer maps each
/// painted output pixel back to an element via the footprint→element ratio:
/// `element = floor((samp_off + delta_px) * tensor_dim / footprint_dim)`. The
/// `row_first`/`col_first` element bounds are the floor at the first painted
/// pixel, so they anchor both the byte fetch and the per-pixel offset.
#[allow(dead_code)]
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
    /// Display footprint of the whole tensor at this tile's zoom level, in
    /// pixels (`disp_w/h * 2^(zoom - max_zoom)`). Drives the pixel→element map.
    pub footprint_w: u64,
    pub footprint_h: u64,
    /// Display-pixel offset, within the tensor's footprint, of the first
    /// painted pixel (`tile_x0`/`tile_y0`).
    pub samp_x0: u64,
    pub samp_y0: u64,
    /// Canvas pixel rectangle this region paints into, in *tile-local* (px, py)
    /// coordinates `[x0, x1) × [y0, y1)`.
    pub tile_x0: u32,
    pub tile_y0: u32,
    pub tile_x1: u32,
    pub tile_y1: u32,
}

/// User-facing layout selection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    /// Auto-select: architectural if every source is safetensors AND
    /// transformer-style names are detected; otherwise hilbert.
    #[default]
    Auto,
    /// Force architectural (structure-aware) layout. Falls back to hilbert if
    /// no source is safetensors.
    Arch,
    /// Force the legacy global-Hilbert layout.
    Hilbert,
}

/// The chosen layout for the current run.
#[allow(dead_code)]
pub enum Layout {
    HilbertGlobal(hilbert::HilbertLayout),
    Architectural(arch::ArchLayout),
}

impl Layout {
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
///
/// `diff_mode` relaxes the all-safetensors gate. In a diff run between two
/// model repos, every per-tensor `TensorDiff` source carries synthetic
/// safetensors info, but non-tensor file diffs (e.g. tokenizer.json) remain
/// without it — under the strict all-safetensors gate they'd block arch
/// selection and force the whole run back to Hilbert. In diff mode we accept
/// arch as long as *some* source has safetensors; non-safetensors diff
/// sources stay in the pipeline but aren't placed on the arch canvas.
pub fn select_layout(
    sources: &[Source],
    cumulative_offsets: &[u64],
    total_bytes: u64,
    mode: LayoutMode,
    metas: &[SourceMeta],
    diff_mode: bool,
) -> Layout {
    // Force-hilbert path: use legacy byte-Hilbert geometry as-is.
    if matches!(mode, LayoutMode::Hilbert) {
        return Layout::HilbertGlobal(hilbert::HilbertLayout::from_total(total_bytes));
    }

    // Architectural requires safetensors data AND a detectable structure. In
    // non-diff mode every source must be safetensors (otherwise the user has
    // explicitly mixed in non-tensor inputs they'd expect to see). In diff
    // mode it's enough that any source carries safetensors info: the typical
    // case is a model-repo diff where the tensor sources are the point and
    // tokenizer/config diffs are incidental.
    let all_safetensors = !sources.is_empty() && sources.iter().all(|s| s.model_info.is_some());
    let any_safetensors = sources.iter().any(|s| s.model_info.is_some());
    let arch_eligible = if diff_mode {
        any_safetensors
    } else {
        all_safetensors
    };

    if arch_eligible {
        if let Some(arch) = arch::ArchLayout::try_build(sources, cumulative_offsets, metas) {
            if diff_mode && !all_safetensors {
                let skipped = sources.iter().filter(|s| s.model_info.is_none()).count();
                log::info!(
                    "arch layout: {skipped} non-safetensors diff source(s) will not appear on the arch canvas (file-level diffs are only rendered in --layout hilbert)"
                );
            }
            return Layout::Architectural(arch);
        }
        if matches!(mode, LayoutMode::Arch) {
            log::warn!(
                "--layout arch requested but no recognisable structure; falling back to hilbert"
            );
        }
    } else if matches!(mode, LayoutMode::Arch) {
        log::warn!(
            "--layout arch requested but no input carries safetensors data; falling back to hilbert"
        );
    }

    Layout::HilbertGlobal(hilbert::HilbertLayout::from_total(total_bytes))
}
