//! Tensor-aware data helpers extracted from `arbvis::data`.
//!
//! These items use `format::*` (Dtype, TensorMeta, etc.) and the
//! safetensors / GGUF / pickle parsers — all of which live in
//! `modelweightvis::format`. arbvis itself stays format-agnostic; it
//! reaches these helpers via the `MoeDiffPrep`, `RepoDiffPrep`,
//! `DirectoryTensorDiffPrep`, and (per-Source) `FormatPlugin` hooks
//! registered by `modelweightvis::register_all`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use futures::stream::{self, StreamExt};
use indicatif::ProgressBar;
use memmap2::Mmap;

use arbvis::hf_url::{self, RemoteFileSpec, RemoteRepo};
use arbvis::xet::XetReader;
use arbvis::{CustomSource, Data, DiffFill, Extensions, Source, SourceKind};

use crate::format::{self, moe::ExpertWeight, ModelInfo, SourceFormat};
use crate::layout::model_config::{ModelConfig, SafetensorsIndex};

/// Bounded concurrency for setup-time HTTP loops. Mirrors arbvis's
/// `SETUP_FETCH_CONCURRENCY` constant; the global throttle is the real cap.
const SETUP_FETCH_CONCURRENCY: usize = 16;

/// Per-tensor RMS sampling concurrency. Smaller than SETUP because the
/// per-call descriptor download (~50–65 MB before dedup) is heavy.
const RMS_SAMPLE_FETCH_CONCURRENCY: usize = 4;

/// Initial header-fetch size for GGUF. Most files fit in 1 MiB; if
/// `Content::read` errors with "unexpected EOF" we retry with the larger
/// size below.
const GGUF_HEADER_FETCH_INITIAL: usize = 1024 * 1024;
const GGUF_HEADER_FETCH_LARGE: usize = 8 * 1024 * 1024;

/// Build a one-shot indicatif progress bar attached to a fresh
/// `MultiProgress` (modelweightvis-local — the moved-from helpers all
/// needed the arbvis-side one which isn't pub-exposed). The pb just
/// emits to stderr; behaviour matches arbvis byte-for-byte modulo the
/// bar group it lives in.
fn setup_progress(label: &str, total: u64) -> Option<ProgressBar> {
    let pb = ProgressBar::new(total);
    pb.set_message(label.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    Some(pb)
}

/// Recursively collect all files under `root`, sorted.
fn collect_files_recursive(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files);
    files.sort();
    files
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("{}: {} — skipping", dir.display(), e);
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_recursive(&path, files);
        }
    }
}

/// Download every remote `Arc<Data>` in `arcs` (those for which `is_local()`
/// returns false) to a local mmap-backed file and swap them in place. Used
/// by the multi-shard tensor diff to amortise xet setup cost. Lifted from
/// arbvis::data unchanged; the xet/HTTP behaviour is identical.
async fn materialize_remote_arcs(
    arcs: &mut [Arc<Data>],
    specs: &[RemoteFileSpec],
    progress_label: &str,
) -> anyhow::Result<()> {
    assert_eq!(
        arcs.len(),
        specs.len(),
        "materialize_remote_arcs: arcs and specs must be the same length"
    );

    let to_download: Vec<(usize, RemoteFileSpec)> = arcs
        .iter()
        .zip(specs.iter())
        .enumerate()
        .filter_map(|(i, (arc, spec))| {
            if arc.is_local() {
                None
            } else {
                Some((i, spec.clone()))
            }
        })
        .collect();

    if to_download.is_empty() {
        return Ok(());
    }

    let indices: Vec<usize> = to_download.iter().map(|(i, _)| *i).collect();
    let specs_subset: Vec<RemoteFileSpec> = to_download.into_iter().map(|(_, s)| s).collect();
    let paths = arbvis::data::download_specs_to_paths(&specs_subset, progress_label).await?;

    for (i, path) in indices.into_iter().zip(paths) {
        let f = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let mmap =
            unsafe { Mmap::map(&f) }.with_context(|| format!("mmap'ing {}", path.display()))?;
        arcs[i] = Arc::new(Data::Mapped(mmap));
    }
    Ok(())
}

// === extracted from arbvis::data (block_aligned_byte_range) ===
/// Block-aligned byte range for reading `len` consecutive elements starting
/// at element `start` from a tensor's data.
///
/// Returns `(byte_offset_from_tensor_start, byte_length, elem_offset_in_buffer)`.
/// For fixed-stride dtypes the element offset is always 0; for block-stride
/// it's the in-block position of `start`, since the buffer must begin on a
/// block boundary so [`crate::format::TensorElementReader`] can decode it.
fn block_aligned_byte_range(dtype: format::Dtype, start: u64, len: u64) -> (u64, usize, usize) {
    use format::ElementStride;
    match dtype.stride() {
        ElementStride::Fixed(bpe) => {
            let bpe = bpe as u64;
            (start * bpe, (len * bpe) as usize, 0)
        }
        ElementStride::Block {
            block_bytes,
            block_elements,
        } => {
            let be = block_elements as u64;
            let bb = block_bytes as u64;
            let first_block = start / be;
            let last_block_excl = (start + len).div_ceil(be);
            let byte_off = first_block * bb;
            let byte_len = (last_block_excl - first_block) * bb;
            let elem_off = (start - first_block * be) as usize;
            (byte_off, byte_len as usize, elem_off)
        }
        // Packed dtypes (AWQ / GPTQ): treat the packed-int slot as the
        // atomic unit. Snap down to the slot containing `start` and up to
        // the slot just past `start + len`; report the in-slot element
        // offset so the reader can skip over `elem_off` elements before
        // returning the requested range. The sidecar scales/zeros buffers
        // are fetched separately by the render path — this function only
        // sizes the qweight read.
        ElementStride::Packed {
            bits,
            pack_dtype_bytes,
            ..
        } => {
            if bits == 0 {
                return (0, 0, 0);
            }
            let elems_per_slot = ((pack_dtype_bytes as u64) * 8) / bits as u64;
            let Some(first_slot) = start.checked_div(elems_per_slot) else {
                return (0, 0, 0);
            };
            let slot_bytes = pack_dtype_bytes as u64;
            let last_slot_excl = (start + len).div_ceil(elems_per_slot);
            let byte_off = first_slot * slot_bytes;
            let byte_len = (last_slot_excl - first_slot) * slot_bytes;
            let elem_off = (start - first_slot * elems_per_slot) as usize;
            (byte_off, byte_len as usize, elem_off)
        }
    }
}

// `LazyFetcher` re-exported from arbvis::data (single source of truth).

// === extracted from arbvis::data (TensorDiffSource_struct,TensorDiffSource_impl) ===
/// Per-tensor diff buffer. The byte stream this exposes is computed lazily
/// from two whole-file `Data` sources whenever the render pipeline calls
/// `fetch_range` on the resulting `Data::LazyDiff`. `metric` selects how
/// per-element deltas are encoded; `scale_orig` carries any per-tensor
/// statistic the metric needs (RMS of `orig` for `DiffMetric::Rms`),
/// pre-computed at setup so the per-tile path stays pure-streaming.
pub struct TensorDiffSource {
    pub orig: Arc<Data>,
    pub mod_: Arc<Data>,
    pub orig_start: u64,
    pub mod_start: u64,
    pub orig_dtype: format::Dtype,
    pub mod_dtype: format::Dtype,
    pub metric: format::DiffMetric,
    pub scale_orig: f32,
    pub byte_size: u64,
}

impl CustomSource for TensorDiffSource {
    fn id(&self) -> &'static str {
        "tensor-diff"
    }

    fn byte_size(&self) -> u64 {
        self.byte_size
    }

    fn open(&self) -> anyhow::Result<Data> {
        let orig = Arc::clone(&self.orig);
        let mod_ = Arc::clone(&self.mod_);
        let orig_start = self.orig_start;
        let mod_start = self.mod_start;
        let orig_dtype = self.orig_dtype;
        let mod_dtype = self.mod_dtype;
        let metric = self.metric;
        let scale_orig = self.scale_orig;
        Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
            let orig = Arc::clone(&orig);
            let mod_ = Arc::clone(&mod_);
            Box::pin(async move {
                // Block-aligned byte ranges: for fixed-stride dtypes
                // these are `start * elem`/`len * elem`; for block-
                // stride they snap down/up to block boundaries and the
                // returned `elem_off` is how many elements into the
                // fetched buffer the requested start element lives.
                let (o_byte_off, o_byte_len, o_elem_off) =
                    block_aligned_byte_range(orig_dtype, start, len as u64);
                let (m_byte_off, m_byte_len, m_elem_off) =
                    block_aligned_byte_range(mod_dtype, start, len as u64);
                let ob = orig
                    .fetch_range(orig_start + o_byte_off, o_byte_len)
                    .await?;
                let mb = mod_.fetch_range(mod_start + m_byte_off, m_byte_len).await?;
                Ok(orig_dtype.diff_to_u8(
                    &ob, o_elem_off, mod_dtype, &mb, m_elem_off, metric, scale_orig, len,
                ))
            })
        })))
    }
}

/// Per-MoE-cell coordinates attached to a `Source.extensions` map when the
/// source represents one expert-vs-expert diff (or self-render diagonal)
/// inside one MoE layer. Presence of this extension is the trigger that
/// routes `select_layout` to the MoE-diff layout plugin.
#[derive(Debug, Clone, Copy)]
pub struct MoeCell {
    pub layer: u32,
    pub weight: ExpertWeight,
    pub i: u32,
    pub j: u32,
}

// === extracted from arbvis::data (load_model_info,fetch_model_header) ===
/// Read just the header of a recognised model file and return parsed
/// metadata. Dispatches on `format`.
pub fn load_model_info(
    path: &Path,
    file_size: u64,
    fmt: SourceFormat,
) -> anyhow::Result<ModelInfo> {
    match fmt {
        SourceFormat::Safetensors => {
            // Read the first 8 bytes to get header_size, then read header_size more bytes.
            let mut f = File::open(path)?;
            let mut size_buf = [0u8; 8];
            f.read_exact(&mut size_buf)?;
            let header_size = u64::from_le_bytes(size_buf);
            if header_size > 100 * 1024 * 1024 {
                anyhow::bail!("header_size={} exceeds 100 MB safety limit", header_size);
            }
            let total_header = 8 + header_size as usize;
            let mut header_buf = vec![0u8; total_header];
            header_buf[..8].copy_from_slice(&size_buf);
            f.read_exact(&mut header_buf[8..])?;
            let (tensors, header_end) = format::safetensors::parse_header(&header_buf)?;
            let color_ranges =
                format::safetensors::build_color_ranges(&tensors, header_end, file_size);
            Ok(ModelInfo {
                format: SourceFormat::Safetensors,
                tensors,
                color_ranges,
            })
        }
        SourceFormat::Gguf => {
            // For GGUF the header size isn't known up-front. Read a 1 MiB
            // prefix first; if parsing fails on EOF, retry with 8 MiB.
            let mut buf = vec![0u8; GGUF_HEADER_FETCH_INITIAL.min(file_size as usize)];
            let mut f = File::open(path)?;
            f.read_exact(&mut buf)?;
            let header = match format::gguf::parse_header(&buf) {
                Ok(h) => h,
                Err(_) if file_size as usize > buf.len() => {
                    let mut bigger = vec![0u8; GGUF_HEADER_FETCH_LARGE.min(file_size as usize)];
                    let mut f = File::open(path)?;
                    f.read_exact(&mut bigger)?;
                    format::gguf::parse_header(&bigger)?
                }
                Err(e) => return Err(e),
            };
            let color_ranges = format::gguf::build_color_ranges(
                &header.tensors,
                header.tensor_data_offset,
                file_size,
            );
            Ok(ModelInfo {
                format: SourceFormat::Gguf,
                tensors: header.tensors,
                color_ranges,
            })
        }
        SourceFormat::Pickle => {
            // Pickle's zip end-of-central-directory lives at the END of the
            // file, so unlike safetensors/GGUF we can't parse from a prefix.
            // candle's pickle reader opens the file by path; pass it through.
            let header = format::pickle::parse_header(path)?;
            let color_ranges = format::pickle::build_color_ranges(
                &header.tensors,
                header.tensor_data_offset,
                file_size,
            );
            Ok(ModelInfo {
                format: SourceFormat::Pickle,
                tensors: header.tensors,
                color_ranges,
            })
        }
    }
}

/// Async sibling of [`load_model_info`] for `Data` sources (Http/Xet/etc.).
/// Wraps [`fetch_model_header`] and builds the same [`ModelInfo`] shape.
///
/// Pickle files cannot be parsed from a remote prefix (the zip
/// end-of-central-directory lives at the tail), so this returns an error
/// for [`SourceFormat::Pickle`] — callers fall back to "treat as plain
/// bytes" the same way they would for an unrecognised file.
pub async fn load_model_info_async(
    data: &Data,
    byte_size: u64,
    fmt: SourceFormat,
) -> anyhow::Result<ModelInfo> {
    let (tensors, header_end) = fetch_model_header(data, fmt).await?;
    let color_ranges = match fmt {
        SourceFormat::Safetensors => {
            format::safetensors::build_color_ranges(&tensors, header_end, byte_size)
        }
        SourceFormat::Gguf => format::gguf::build_color_ranges(&tensors, header_end, byte_size),
        SourceFormat::Pickle => format::pickle::build_color_ranges(&tensors, header_end, byte_size),
    };
    Ok(ModelInfo {
        format: fmt,
        tensors,
        color_ranges,
    })
}

/// Fetch and parse the header from any `Data` source. For local sources
/// (Mapped/Owned) the fetches are zero-copy slices. For remote sources
/// (Http/Xet) this issues one or two range requests.
async fn fetch_model_header(
    data: &Data,
    fmt: SourceFormat,
) -> anyhow::Result<(Vec<format::TensorMeta>, u64)> {
    match fmt {
        SourceFormat::Safetensors => {
            let size_bytes = data.fetch_range(0, 8).await?;
            let header_size = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
            if header_size > 100 * 1024 * 1024 {
                anyhow::bail!(
                    "safetensors header_size={} exceeds 100 MB safety limit",
                    header_size
                );
            }
            let total_header = 8 + header_size as usize;
            let header_bytes = data.fetch_range(0, total_header).await?;
            format::safetensors::parse_header(&header_bytes)
        }
        SourceFormat::Gguf => {
            let buf = data.fetch_range(0, GGUF_HEADER_FETCH_INITIAL).await?;
            let header = match format::gguf::parse_header(&buf) {
                Ok(h) => h,
                Err(_) => {
                    let bigger = data.fetch_range(0, GGUF_HEADER_FETCH_LARGE).await?;
                    format::gguf::parse_header(&bigger)?
                }
            };
            Ok((header.tensors, header.tensor_data_offset))
        }
        SourceFormat::Pickle => {
            // The zip end-of-central-directory record lives at the END of a
            // pickle file (.bin / .pth / .pt), so we can't parse from a head
            // prefix the way we do for safetensors/GGUF. Supporting this
            // would need a tail-first range fetch plus an in-memory zip
            // parser; for v1 we surface a clear error and let the caller
            // fall back to "treat as plain bytes".
            anyhow::bail!("pickle: remote header fetch not yet supported — download the file first")
        }
    }
}

// === extracted from arbvis::data (prepare_diff_sources_from_http,strip_prefix_components,find_strip_depths,TensorMatch,match_under_strip_depths,find_matched_tensor_pairs,fetch_rms_estimates,build_multi_safetensors_diff_sources_inner,build_multi_safetensors_diff_sources,build_multi_safetensors_diff_sources_from_http,prepare_moe_diff_sources,build_safetensors_diff_sources,SourceMeta,try_load_source_meta,load_meta_for_sources,fetch_hf_sidecar) ===
/// Build diff sources from two repos listed as HTTP specs (no download).
///
/// Safetensors files are diffed lazily via range requests — no model weights
/// are downloaded to disk or held in RAM. Small non-safetensors files (≤16 MB)
/// are downloaded eagerly and binary-diffed; larger ones are skipped with a warning.
pub async fn prepare_diff_sources_from_http(
    orig_specs: &[(String, RemoteFileSpec)],
    mod_specs: &[(String, RemoteFileSpec)],
    is_finetune: bool,
    metric: format::DiffMetric,
    stream: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let is_st = |name: &str| name.ends_with(".safetensors");

    let orig_st: Vec<&(String, RemoteFileSpec)> =
        orig_specs.iter().filter(|(n, _)| is_st(n)).collect();
    let mod_st: Vec<&(String, RemoteFileSpec)> =
        mod_specs.iter().filter(|(n, _)| is_st(n)).collect();

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    // Safetensors diff — disk-backed by default; pass `stream=true` to keep
    // shards remote and diff each tile via HTTP range requests instead.
    if !orig_st.is_empty() || !mod_st.is_empty() {
        match build_multi_safetensors_diff_sources_from_http(
            &orig_st,
            &mod_st,
            is_finetune,
            metric,
            stream,
        )
        .await
        {
            Ok((mut tensor_sources, bytes)) => {
                let base_idx = sources.len();
                for s in &mut tensor_sources {
                    s.file_idx += base_idx;
                }
                sources.extend(tensor_sources);
                total += bytes;
            }
            Err(e) => log::warn!("safetensors diff failed: {e} — skipping"),
        }
    }

    // Non-safetensors files: match by filename. Same-size pairs become a byte
    // diff (downloaded if small); different-size or single-side files become
    // crosshatched unmatched regions so they remain visible. Large files are
    // sized but rendered as the orig-fill kind (we can't byte-diff something
    // we won't download).
    const MAX_EAGER_SIZE: u64 = 16 * 1024 * 1024;
    let orig_fill_kind = if is_finetune {
        DiffFill::Grey
    } else {
        DiffFill::Red
    };
    let orig_non: HashMap<&str, &RemoteFileSpec> = orig_specs
        .iter()
        .filter(|(n, _)| !is_st(n))
        .map(|(n, s)| (n.as_str(), s))
        .collect();
    let mod_non: HashMap<&str, &RemoteFileSpec> = mod_specs
        .iter()
        .filter(|(n, _)| !is_st(n))
        .map(|(n, s)| (n.as_str(), s))
        .collect();

    let mut mod_only_files: Vec<&str> = mod_non
        .keys()
        .copied()
        .filter(|k| !orig_non.contains_key(k))
        .collect();
    mod_only_files.sort();
    if is_finetune && !mod_only_files.is_empty() {
        log::warn!(
            "--diff --finetune: modified side has {} file(s) with no counterpart on the \
             original/base side — rendering as green crosshatch: {}",
            mod_only_files.len(),
            mod_only_files.join(", ")
        );
    }

    let mut sorted: Vec<&str> = orig_non.keys().copied().collect();
    sorted.sort();

    // First pass (sync): partition into byte-diff jobs vs unmatched-region
    // sources. Diff jobs are downloaded in parallel afterwards.
    let mut diff_jobs: Vec<(String, RemoteFileSpec, RemoteFileSpec)> = Vec::new();
    let mut unmatched_orig: Vec<(String, u64, DiffFill)> = Vec::new();
    let mut unmatched_mod: Vec<(String, u64, DiffFill)> = Vec::new();
    // Files present in both repos but too large to eagerly byte-diff. Rendered
    // as a single crosshatch source per file (not split across both sides).
    let mut unmatched_modified: Vec<(String, u64, DiffFill)> = Vec::new();
    for fname in sorted {
        let orig_spec = &orig_non[fname];
        let mod_spec = match mod_non.get(fname) {
            Some(s) => s,
            None => {
                if orig_spec.size > 0 {
                    unmatched_orig.push((fname.to_string(), orig_spec.size, orig_fill_kind));
                }
                continue;
            }
        };
        if orig_spec.size != mod_spec.size {
            if is_finetune {
                log::warn!(
                    "--diff --finetune: size mismatch for {fname} ({} vs {} bytes) — \
                     byte-diffing with zero-padding on the shorter side",
                    orig_spec.size,
                    mod_spec.size
                );
            } else {
                log::warn!(
                    "size mismatch for {fname} ({} vs {} bytes) — byte-diffing with zero-padding",
                    orig_spec.size,
                    mod_spec.size
                );
            }
        }
        let max_size = orig_spec.size.max(mod_spec.size);
        if max_size == 0 {
            continue;
        }
        if max_size > MAX_EAGER_SIZE {
            // Too large to byte-diff eagerly; surface as a single crosshatched
            // region sized to max(orig, mod) so the file appears once in the
            // canvas, regardless of whether sizes match.
            log::warn!(
                "{fname} exceeds {} MB — rendering as crosshatched region instead of byte diff",
                MAX_EAGER_SIZE / 1024 / 1024
            );
            unmatched_modified.push((fname.to_string(), max_size, orig_fill_kind));
            continue;
        }
        diff_jobs.push((fname.to_string(), (*orig_spec).clone(), (*mod_spec).clone()));
    }
    for fname in &mod_only_files {
        let spec = &mod_non[fname];
        if spec.size > 0 {
            unmatched_mod.push((fname.to_string(), spec.size, DiffFill::Green));
        }
    }

    let pb = setup_progress(
        "file pairs (non-safetensors diff downloads)",
        diff_jobs.len() as u64,
    );
    let pb_for_workers = pb.clone();
    let diffs: Vec<anyhow::Result<(String, Vec<u8>)>> = stream::iter(diff_jobs)
        .map(|(fname, orig_spec, mod_spec)| {
            let pb = pb_for_workers.clone();
            async move {
                let orig_data = Data::Http {
                    repo: orig_spec.repo.clone(),
                    filename: Arc::clone(&orig_spec.filename),
                    revision: Arc::clone(&orig_spec.revision),
                };
                let mod_data = Data::Http {
                    repo: mod_spec.repo.clone(),
                    filename: Arc::clone(&mod_spec.filename),
                    revision: Arc::clone(&mod_spec.revision),
                };
                let ob = orig_data.fetch_range(0, orig_spec.size as usize).await?;
                let mb = mod_data.fetch_range(0, mod_spec.size as usize).await?;
                // Pad the shorter side with zeros so size-mismatched but
                // same-named files share one diff source. The longer side's
                // tail diffs against zero, which renders as deltas indicating
                // bytes that exist on only one side.
                let len = ob.len().max(mb.len());
                let diff: Vec<u8> = (0..len)
                    .map(|i| {
                        let a = ob.get(i).copied().unwrap_or(0);
                        let b = mb.get(i).copied().unwrap_or(0);
                        let delta = b as i16 - a as i16;
                        let brightness =
                            (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                        if delta >= 0 {
                            127u8 + brightness
                        } else {
                            127u8 - brightness
                        }
                    })
                    .collect();
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                Ok((fname, diff))
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }
    // Re-sort by filename so the Source order is deterministic.
    let mut diffs: Vec<(String, Vec<u8>)> =
        diffs.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
    diffs.sort_by(|a, b| a.0.cmp(&b.0));
    for (fname, diff) in diffs {
        let size = diff.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(diff),
            byte_size: size,
            name_override: Some(fname),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }

    // Unmatched / oversize / size-mismatch files surface as crosshatched
    // regions so the user sees they exist even though no byte diff was
    // computed.
    unmatched_orig.sort();
    for (fname, size, fill) in unmatched_orig {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            name_override: Some(format!("[only in original] {fname}")),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }
    unmatched_mod.sort();
    for (fname, size, fill) in unmatched_mod {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            name_override: Some(format!("[only in modified] {fname}")),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }
    unmatched_modified.sort();
    for (fname, size, fill) in unmatched_modified {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            name_override: Some(fname),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }

    if sources.is_empty() {
        anyhow::bail!("--diff: no matching file pairs found between the two repos");
    }
    Ok((sources, total))
}

/// Strip the first `n` dot-delimited path components from `name`.
/// Returns the remainder after the n-th dot, or `None` if `name` has fewer than n+1 components.
fn strip_prefix_components(name: &str, n: usize) -> Option<&str> {
    let mut idx = 0;
    for _ in 0..n {
        idx += name[idx..].find('.')? + 1;
    }
    Some(&name[idx..])
}

/// Find (strip_orig, strip_mod) prefix depths that maximise unique 1-to-1 tensor name matches.
/// Returns (0, 0) if exact matching already produces matches.
fn find_strip_depths(orig_names: &[String], mod_names: &[String]) -> (usize, usize) {
    // Count unique-suffix occurrences for a set of names stripped by `n` components.
    let unique_suffixes = |names: &[String], n: usize| -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for name in names {
            if let Some(s) = strip_prefix_components(name, n) {
                if !s.is_empty() {
                    *counts.entry(s.to_owned()).or_insert(0) += 1;
                }
            }
        }
        counts
    };

    let mut best = (0usize, 0usize, 0usize); // (strip_orig, strip_mod, match_count)
    for so in 0usize..=5 {
        let orig_counts = unique_suffixes(orig_names, so);
        for sm in 0usize..=5 {
            if so == 0 && sm == 0 {
                continue;
            }
            let mod_counts = unique_suffixes(mod_names, sm);
            let matches = orig_counts
                .iter()
                .filter(|(s, &oc)| oc == 1 && mod_counts.get(s.as_str()) == Some(&1))
                .count();
            if matches > best.2 {
                best = (so, sm, matches);
            }
        }
    }
    (best.0, best.1)
}

/// Result of tensor-name matching across the two sides of a diff.
pub struct TensorMatch {
    /// 1-to-1 matched pairs `(orig_full, mod_full)`, sorted by `orig_full`.
    pub pairs: Vec<(String, String)>,
    /// Tensor full names present only on the original side, sorted.
    pub orig_only: Vec<String>,
    /// Tensor full names present only on the modified side, sorted.
    pub mod_only: Vec<String>,
}

/// Match tensors under a fixed `(strip_o, strip_m)` strip pair: build the
/// stripped-suffix maps (with collisions blanked), then pair up unique 1-to-1
/// matches. Returns the matched `(orig_full, mod_full)` pairs only — caller
/// drives iteration and tracks the unmatched residual.
fn match_under_strip_depths(
    orig_names: &[String],
    mod_names: &[String],
    strip_o: usize,
    strip_m: usize,
) -> Vec<(String, String)> {
    let orig_by_stripped: HashMap<String, &str> = orig_names
        .iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_o)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mod_by_stripped: HashMap<String, &str> = mod_names
        .iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_m)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mut sorted_orig: Vec<&str> = orig_by_stripped
        .values()
        .copied()
        .filter(|s| !s.is_empty())
        .collect();
    sorted_orig.sort();

    let mut pairs = Vec::new();
    for orig_full in sorted_orig {
        let stripped = match strip_prefix_components(orig_full, strip_o) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        if let Some(&mod_full) = mod_by_stripped.get(stripped) {
            if !mod_full.is_empty() {
                pairs.push((orig_full.to_owned(), mod_full.to_owned()));
            }
        }
    }
    pairs
}

/// Find matched + unmatched tensor name groupings between two name sets.
///
/// **Multi-pass strip heuristic.** Real-world model files frequently mix
/// multiple wrapper-induced prefix nestings — e.g. GRaPE-2-Nano's language
/// tensors live under `model.language_model.language_model.language_model.*`
/// (matching the base's `model.language_model.*` at strip depths `(1, 3)`)
/// while its vision tensors live under `model.language_model.visual.*`
/// (matching the base's `model.visual.*` at strip depths `(2, 3)`). A single
/// `(strip_o, strip_m)` pair can't capture both — so we iterate:
///
/// 1. Pull out all exact-name matches first.
/// 2. Greedily pick the best `(strip_o, strip_m)` over the remaining
///    unmatched tensors, apply those matches, repeat.
/// 3. Stop when no further pair yields any matches.
///
/// Unmatched tensors are returned so callers can surface them (e.g. as
/// crosshatch fill) rather than silently dropping them.
fn find_matched_tensor_pairs(orig_names: &[String], mod_names: &[String]) -> TensorMatch {
    use std::collections::HashSet;
    let mut remaining_orig: HashSet<String> = orig_names.iter().cloned().collect();
    let mut remaining_mod: HashSet<String> = mod_names.iter().cloned().collect();

    let mut pairs: Vec<(String, String)> = Vec::new();

    // Pass 0: exact-name overlap. Iterate the input order so the resulting
    // log line is deterministic on repeat runs.
    for name in orig_names {
        if remaining_orig.contains(name) && remaining_mod.contains(name) {
            remaining_orig.remove(name);
            remaining_mod.remove(name);
            pairs.push((name.clone(), name.clone()));
        }
    }
    if !pairs.is_empty() {
        log::debug!(
            "safetensors diff: pass 0 exact match — {} pairs",
            pairs.len()
        );
    }

    // Subsequent passes: greedy multi-pass strip search. Each pass picks the
    // (strip_o, strip_m) that yields the most matches over the *remaining*
    // unmatched tensors, applies those matches, and continues. Bounded by
    // the strip search range (find_strip_depths) and by termination once no
    // pair yields any matches.
    let mut pass = 0usize;
    loop {
        if remaining_orig.is_empty() || remaining_mod.is_empty() {
            break;
        }
        let orig_vec: Vec<String> = remaining_orig.iter().cloned().collect();
        let mod_vec: Vec<String> = remaining_mod.iter().cloned().collect();
        let (strip_o, strip_m) = find_strip_depths(&orig_vec, &mod_vec);
        if strip_o == 0 && strip_m == 0 {
            break;
        }
        let new_pairs = match_under_strip_depths(&orig_vec, &mod_vec, strip_o, strip_m);
        if new_pairs.is_empty() {
            break;
        }
        pass += 1;
        log::info!(
            "safetensors diff: strip-match pass {}: stripping {} component(s) from original and {} from modified — {} new pair(s)",
            pass, strip_o, strip_m, new_pairs.len()
        );
        for (o, m) in &new_pairs {
            remaining_orig.remove(o);
            remaining_mod.remove(m);
        }
        pairs.extend(new_pairs);
    }

    let mut orig_only: Vec<String> = remaining_orig.into_iter().collect();
    let mut mod_only: Vec<String> = remaining_mod.into_iter().collect();
    orig_only.sort();
    mod_only.sort();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    TensorMatch {
        pairs,
        orig_only,
        mod_only,
    }
}

/// Sample each matched orig tensor's bytes and compute its RMS, used as the
/// per-tensor scale for `DiffMetric::Rms`. A 64 KB contiguous prefix is more
/// than enough for a stable estimate; for HTTP/Xet sources this is one extra
/// range fetch per matched tensor at setup, parallelised over the throttle.
async fn fetch_rms_estimates(
    paired_ok: &[(String, String)],
    orig_map: &HashMap<String, (usize, format::TensorMeta)>,
    orig_data: &[Arc<Data>],
) -> anyhow::Result<Vec<f32>> {
    const SCALE_SAMPLE_BYTES: u64 = 64 * 1024;
    let pb = setup_progress("orig tensor RMS samples", paired_ok.len() as u64);
    let inputs: Vec<(usize, usize, u64, u64, format::Dtype)> = paired_ok
        .iter()
        .enumerate()
        .map(|(idx, (orig_full, _))| {
            let (oi, orig_t) = &orig_map[orig_full];
            let elem = orig_t.dtype.element_size() as u64;
            let tensor_bytes = orig_t.file_end.saturating_sub(orig_t.file_start);
            let want = SCALE_SAMPLE_BYTES.min(tensor_bytes);
            // Align sample length down to a whole element so rms_from_buf
            // only sees complete values.
            let len = want.checked_div(elem).map(|q| q * elem).unwrap_or(0);
            (idx, *oi, orig_t.file_start, len, orig_t.dtype)
        })
        .collect();
    let mut out: Vec<(usize, f32)> = stream::iter(inputs)
        .map(|(idx, oi, start, len, dtype)| {
            let d = Arc::clone(&orig_data[oi]);
            let pb = pb.clone();
            async move {
                let scale = if len == 0 {
                    0.0
                } else {
                    match d.fetch_range(start, len as usize).await {
                        Ok(bytes) => format::rms_from_buf(dtype, &bytes),
                        Err(e) => {
                            log::warn!("safetensors diff: orig RMS sample failed ({e}); using 0.0");
                            0.0
                        }
                    }
                };
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                (idx, scale)
            }
        })
        .buffer_unordered(RMS_SAMPLE_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }
    out.sort_by_key(|(i, _)| *i);
    Ok(out.into_iter().map(|(_, s)| s).collect())
}

/// Core tensor diff builder: given two parallel lists of whole-file Data sources,
/// build per-tensor TensorDiff Source entries without reading any tensor bytes.
async fn build_multi_safetensors_diff_sources_inner(
    orig_data: &[Arc<Data>],
    orig_fmts: &[SourceFormat],
    mod_data: &[Arc<Data>],
    mod_fmts: &[SourceFormat],
    is_finetune: bool,
    metric: format::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Two HTTP range requests per file (8-byte preamble + variable header
    // for safetensors; one or two prefix fetches for GGUF). For sharded
    // models this is dozens of files per side; serializing them wastes the
    // throttle.
    let total = (orig_data.len() + mod_data.len()) as u64;
    let pb = setup_progress("source files (model headers)", total);

    async fn fetch_all(
        data: &[Arc<Data>],
        fmts: &[SourceFormat],
        side: &'static str,
        pb: &Option<ProgressBar>,
    ) -> anyhow::Result<Vec<(usize, Vec<format::TensorMeta>)>> {
        let pb = pb.clone();
        // Collect owned `(Arc<Data>, SourceFormat)` pairs up front so the
        // per-item closure below doesn't borrow from `data`/`fmts` — that
        // borrow's lifetime is too narrow to satisfy the higher-ranked
        // bound required when this helper is reached from
        // `DiffSourceBuilder::try_build`'s async-trait future.
        let items: Vec<(Arc<Data>, SourceFormat)> =
            data.iter().cloned().zip(fmts.iter().copied()).collect();
        let mut out: Vec<(usize, anyhow::Result<Vec<format::TensorMeta>>)> =
            stream::iter(items.into_iter().enumerate())
                .map(|(i, (d, fmt))| {
                    let pb = pb.clone();
                    async move {
                        let r = fetch_model_header(&d, fmt)
                            .await
                            .map(|(t, _)| t)
                            .with_context(|| format!("reading {fmt:?} header for {side} file {i}"));
                        if let Some(pb) = pb.as_ref() {
                            pb.inc(1);
                        }
                        (i, r)
                    }
                })
                .buffer_unordered(SETUP_FETCH_CONCURRENCY)
                .collect()
                .await;
        out.sort_by_key(|(i, _)| *i);
        out.into_iter().map(|(i, r)| r.map(|t| (i, t))).collect()
    }

    let orig_headers = fetch_all(orig_data, orig_fmts, "orig", &pb).await?;
    let mod_headers = fetch_all(mod_data, mod_fmts, "mod", &pb).await?;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    // Use the first source's format on each side for name canonicalisation.
    // In practice all files on one side of a diff share a format; the
    // canonical name table is identity for safetensors and translates GGUF
    // tensor names to their HF equivalents.
    let orig_canon_fmt = orig_fmts
        .first()
        .copied()
        .unwrap_or(SourceFormat::Safetensors);
    let mod_canon_fmt = mod_fmts
        .first()
        .copied()
        .unwrap_or(SourceFormat::Safetensors);

    let mut orig_map: HashMap<String, (usize, format::TensorMeta)> = HashMap::new();
    for (i, tensors) in orig_headers {
        for t in tensors {
            orig_map.entry(t.name.clone()).or_insert((i, t));
        }
    }
    let mut mod_map: HashMap<String, (usize, format::TensorMeta)> = HashMap::new();
    for (i, tensors) in mod_headers {
        for t in tensors {
            mod_map.entry(t.name.clone()).or_insert((i, t));
        }
    }

    // Canonical name table: maps raw → canonical for each side.
    let orig_canon: HashMap<String, String> = orig_map
        .keys()
        .map(|n| (n.clone(), orig_canon_fmt.canonical_name(n)))
        .collect();
    let mod_canon: HashMap<String, String> = mod_map
        .keys()
        .map(|n| (n.clone(), mod_canon_fmt.canonical_name(n)))
        .collect();
    // Reverse maps: canonical → raw, so after matching on canonical names we
    // can look the original name back up.
    let orig_by_canon: HashMap<String, String> = orig_canon
        .iter()
        .map(|(raw, can)| (can.clone(), raw.clone()))
        .collect();
    let mod_by_canon: HashMap<String, String> = mod_canon
        .iter()
        .map(|(raw, can)| (can.clone(), raw.clone()))
        .collect();
    let orig_canon_names: Vec<String> = orig_by_canon.keys().cloned().collect();
    let mod_canon_names: Vec<String> = mod_by_canon.keys().cloned().collect();
    let TensorMatch {
        pairs: canon_pairs,
        orig_only: canon_orig_only,
        mod_only: canon_mod_only,
    } = find_matched_tensor_pairs(&orig_canon_names, &mod_canon_names);
    // Translate back to raw names for the rest of the pipeline.
    let pairs: Vec<(String, String)> = canon_pairs
        .into_iter()
        .filter_map(|(co, cm)| {
            let o = orig_by_canon.get(&co)?.clone();
            let m = mod_by_canon.get(&cm)?.clone();
            Some((o, m))
        })
        .collect();
    let orig_only: Vec<String> = canon_orig_only
        .into_iter()
        .filter_map(|c| orig_by_canon.get(&c).cloned())
        .collect();
    let mod_only: Vec<String> = canon_mod_only
        .into_iter()
        .filter_map(|c| mod_by_canon.get(&c).cloned())
        .collect();
    let _ = (&orig_canon, &mod_canon); // intentional: keep maps alive above

    // Tensors present in both, but with incompatible shapes, can't be diffed
    // element-wise. Treat each side independently: in non-finetune mode both
    // sides surface as unmatched (red on orig, green on mod). In finetune
    // mode the modified side is an error (see below), so we fail fast.
    let mut shape_mismatch: Vec<(String, String)> = Vec::new();
    let mut paired_ok: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    for (orig_full, mod_full) in pairs {
        let orig_t = &orig_map[&orig_full].1;
        let mod_t = &mod_map[&mod_full].1;
        if orig_t.shape != mod_t.shape {
            shape_mismatch.push((orig_full, mod_full));
        } else {
            paired_ok.push((orig_full, mod_full));
        }
    }

    // Finetune contract: every tensor the finetune ships should exist (with
    // the same shape) on the base side. Real-world models sometimes break
    // this — e.g. wrapper-saved finetunes that nest a vision tower under a
    // language_model prefix the base doesn't share — so we warn rather than
    // bail, and surface the offending tensors as green crosshatch via the
    // normal only-in-modified rendering below.
    if is_finetune {
        let mut mod_extras: Vec<String> = mod_only.clone();
        for (_, mod_full) in &shape_mismatch {
            mod_extras.push(mod_full.clone());
        }
        if !mod_extras.is_empty() {
            mod_extras.sort();
            log::warn!(
                "safetensors diff --finetune: modified side has {} tensor(s) not present \
                 (or with mismatched shape) on the original/base side — rendering as green \
                 crosshatch: {}",
                mod_extras.len(),
                mod_extras.join(", ")
            );
        }
    }

    // For DiffMetric::Rms we need a per-tensor scale (RMS of orig). Sample
    // up to RMS_SAMPLE_ELEMS elements per tensor via a single range fetch;
    // for HTTP sources this is one extra request per tensor at setup time,
    // for local mmap it's free. AbsLog and Exact don't need a scale.
    let scales: Vec<f32> = if matches!(metric, format::DiffMetric::Rms) {
        fetch_rms_estimates(&paired_ok, &orig_map, orig_data).await?
    } else {
        vec![0.0; paired_ok.len()]
    };

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    for ((orig_full, mod_full), scale_orig) in paired_ok.iter().zip(scales.iter()) {
        let (oi, orig_t) = &orig_map[orig_full];
        let (mi, mod_t) = &mod_map[mod_full];

        let nelem: u64 = orig_t.shape.iter().product();
        // Describe the diff buffer as a synthetic 1-byte-per-element tensor so
        // the architectural layout can place it at its natural 2D shape. The
        // dtype is U8 (the output of `Dtype::diff_to_u8`); the *element shape*
        // tracks the original tensor's so layer-N q_proj stacks pixel-aligned
        // with layer-N+1 q_proj.
        let diff_meta = format::TensorMeta {
            name: orig_t.name.clone(),
            dtype: format::Dtype::U8,
            shape: orig_t.shape.clone(),
            file_start: 0,
            file_end: nelem,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![diff_meta],
            color_ranges: Vec::new(),
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Custom(Box::new(TensorDiffSource {
                orig: Arc::clone(&orig_data[*oi]),
                mod_: Arc::clone(&mod_data[*mi]),
                orig_start: orig_t.file_start,
                mod_start: mod_t.file_start,
                orig_dtype: orig_t.dtype,
                mod_dtype: mod_t.dtype,
                metric,
                scale_orig: *scale_orig,
                byte_size: nelem,
            })),
            byte_size: nelem,
            name_override: Some(orig_t.label()),
            xet_terms: None,
            extensions,
        });
        total += nelem;
    }

    // Unmatched / shape-mismatched tensors become crosshatched canvas regions.
    // In finetune mode only orig-only entries can survive (mod-side errors
    // were already raised above), and they render as informational grey.
    let orig_fill = if is_finetune {
        DiffFill::Grey
    } else {
        DiffFill::Red
    };

    let mut orig_unmatched: Vec<&format::TensorMeta> = Vec::new();
    for name in &orig_only {
        orig_unmatched.push(&orig_map[name].1);
    }
    for (orig_full, _) in &shape_mismatch {
        orig_unmatched.push(&orig_map[orig_full].1);
    }
    for t in orig_unmatched {
        let nelem: u64 = t.shape.iter().product();
        if nelem == 0 {
            continue;
        }
        // Carry the original shape so the gate in `select_layout` treats this
        // as a safetensors-aware source. The arch layout currently *skips*
        // UnmatchedRegion entries (they're drawn via the crosshatch overlay in
        // the Hilbert path); attaching the synthetic meta keeps future arch
        // crosshatch wiring straightforward.
        let unmatched_meta = format::TensorMeta {
            name: t.name.clone(),
            dtype: format::Dtype::U8,
            shape: t.shape.clone(),
            file_start: 0,
            file_end: nelem,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![unmatched_meta],
            color_ranges: Vec::new(),
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill: orig_fill },
            byte_size: nelem,
            name_override: Some(format!("[only in original] {}", t.label())),
            xet_terms: None,
            extensions,
        });
        total += nelem;
    }

    // mod-only tensors render as green crosshatch in both modes. In finetune
    // mode the warning above already flagged the contract violation; the
    // green crosshatch surfaces it visually too.
    for name in &mod_only {
        let t = &mod_map[name].1;
        let nelem: u64 = t.shape.iter().product();
        if nelem == 0 {
            continue;
        }
        let unmatched_meta = format::TensorMeta {
            name: t.name.clone(),
            dtype: format::Dtype::U8,
            shape: t.shape.clone(),
            file_start: 0,
            file_end: nelem,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![unmatched_meta],
            color_ranges: Vec::new(),
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion {
                fill: DiffFill::Green,
            },
            byte_size: nelem,
            name_override: Some(format!("[only in modified] {}", t.label())),
            xet_terms: None,
            extensions,
        });
        total += nelem;
    }

    if !orig_only.is_empty() || !mod_only.is_empty() || !shape_mismatch.is_empty() {
        log::info!(
            "safetensors diff: {} matched, {} only in original, {} only in modified, {} shape-mismatch",
            sources
                .iter()
                .filter(|s| matches!(&s.kind, SourceKind::Custom(c) if c.id() == "tensor-diff"))
                .count(),
            orig_only.len(),
            mod_only.len(),
            shape_mismatch.len()
        );
    }

    Ok((sources, total))
}

/// Build per-tensor diff Sources from multiple local model files on each
/// side. Each path's extension determines its format (`.safetensors` or
/// `.gguf`); mixed-format pairs are routed through the cross-format name
/// canonicaliser.
pub async fn build_multi_safetensors_diff_sources(
    orig_files: &[PathBuf],
    mod_files: &[PathBuf],
    is_finetune: bool,
    metric: format::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let open_arcs = |files: &[PathBuf]| -> anyhow::Result<(Vec<Arc<Data>>, Vec<SourceFormat>)> {
        let mut datas = Vec::with_capacity(files.len());
        let mut fmts = Vec::with_capacity(files.len());
        for p in files {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            datas.push(Arc::new(Data::Mapped(unsafe { Mmap::map(&f) }?)));
            fmts.push(SourceFormat::from_path(p).unwrap_or(SourceFormat::Safetensors));
        }
        Ok((datas, fmts))
    };
    let (orig_data, orig_fmts) = open_arcs(orig_files)?;
    let (mod_data, mod_fmts) = open_arcs(mod_files)?;
    build_multi_safetensors_diff_sources_inner(
        &orig_data,
        &orig_fmts,
        &mod_data,
        &mod_fmts,
        is_finetune,
        metric,
    )
    .await
}

/// Build per-tensor diff Sources from multiple remote .safetensors files on each side.
///
/// When `stream` is false (the default) every shard is downloaded to the
/// local hf-hub cache and mmapped before any source is constructed, so the
/// per-tile diff path is pure memcpy. When `stream` is true each xet-backed
/// file gets a `Data::Xet` reader (one V2 reconstruction fetch per file,
/// then direct-CAS range fetches afterward — see `XetReader`) and non-xet
/// remote files fall back to `Data::Http`; every per-tile diff then issues
/// an HTTP range request.
async fn build_multi_safetensors_diff_sources_from_http(
    orig_specs: &[&(String, RemoteFileSpec)],
    mod_specs: &[&(String, RemoteFileSpec)],
    is_finetune: bool,
    metric: format::DiffMetric,
    stream: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_fmts: Vec<SourceFormat> = orig_specs
        .iter()
        .map(|(n, _)| SourceFormat::from_name(n).unwrap_or(SourceFormat::Safetensors))
        .collect();
    let mod_fmts: Vec<SourceFormat> = mod_specs
        .iter()
        .map(|(n, _)| SourceFormat::from_name(n).unwrap_or(SourceFormat::Safetensors))
        .collect();
    // Streaming arc construction: per-spec xet reconstruction + Http fallback.
    // Only used when `stream` is true; the disk-backed branch skips it and
    // goes straight through `materialize_remote_arcs`.
    async fn make_streaming_arcs(specs: Vec<RemoteFileSpec>) -> anyhow::Result<Vec<Arc<Data>>> {
        let total = specs.len() as u64;
        let pb = setup_progress("source files (xet reconstruction for diff)", total);
        let pb_for_workers = pb.clone();
        let mut out: Vec<(usize, anyhow::Result<Arc<Data>>)> = stream::iter(specs.into_iter().enumerate())
            .map(|(i, spec)| {
                let pb = pb_for_workers.clone();
                async move {
                    let r: anyhow::Result<Arc<Data>> = if spec.xet_hash.is_some() {
                        match XetReader::new(&spec).await {
                            Ok(reader) => Ok(Arc::new(Data::Xet(reader))),
                            Err(e) => {
                                log::warn!(
                                    "{}: XetReader build failed ({e}); falling back to hf-hub Data::Http",
                                    spec.filename,
                                );
                                Ok(Arc::new(Data::Http {
                                    repo: spec.repo.clone(),
                                    filename: Arc::clone(&spec.filename),
                                    revision: Arc::clone(&spec.revision),
                                }))
                            }
                        }
                    } else {
                        Ok(Arc::new(Data::Http {
                            repo: spec.repo.clone(),
                            filename: Arc::clone(&spec.filename),
                            revision: Arc::clone(&spec.revision),
                        }))
                    };
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (i, r)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;
        if let Some(pb) = pb.as_ref() {
            pb.finish_and_clear();
        }
        out.sort_by_key(|(i, _)| *i);
        out.into_iter()
            .map(|(_, r)| r)
            .collect::<anyhow::Result<Vec<_>>>()
    }
    let orig_specs_owned: Vec<RemoteFileSpec> = orig_specs.iter().map(|(_, s)| s.clone()).collect();
    let mod_specs_owned: Vec<RemoteFileSpec> = mod_specs.iter().map(|(_, s)| s.clone()).collect();
    let (mut orig_data, mut mod_data) = if stream {
        let orig_data = make_streaming_arcs(orig_specs_owned.clone()).await?;
        let mod_data = make_streaming_arcs(mod_specs_owned.clone()).await?;
        (orig_data, mod_data)
    } else {
        // Placeholder arcs swapped in place by materialize_remote_arcs.
        let placeholder = |specs: &[RemoteFileSpec]| -> Vec<Arc<Data>> {
            specs
                .iter()
                .map(|spec| {
                    Arc::new(Data::Http {
                        repo: spec.repo.clone(),
                        filename: Arc::clone(&spec.filename),
                        revision: Arc::clone(&spec.revision),
                    })
                })
                .collect()
        };
        (
            placeholder(&orig_specs_owned),
            placeholder(&mod_specs_owned),
        )
    };
    if !stream {
        materialize_remote_arcs(
            &mut orig_data,
            &orig_specs_owned,
            "source files (downloading for diff: orig side)",
        )
        .await?;
        materialize_remote_arcs(
            &mut mod_data,
            &mod_specs_owned,
            "source files (downloading for diff: modified side)",
        )
        .await?;
    }
    build_multi_safetensors_diff_sources_inner(
        &orig_data,
        &orig_fmts,
        &mod_data,
        &mod_fmts,
        is_finetune,
        metric,
    )
    .await
}

/// Build the per-expert N×N diff matrix sources for a single MoE checkpoint.
///
/// Each routed-experts MLP layer expands into `3 * N * (N + 1) / 2` Sources:
/// for every weight in `{gate_proj, up_proj, down_proj}` and every expert
/// pair `(i, j)` with `i <= j`, one `SourceKind::TensorDiff` whose two byte
/// ranges point into the *same* checkpoint at experts `i` and `j`. Diagonal
/// cells (`i == j`) point both sides at the same offset, so under any metric
/// they paint as a uniform "no change" colour — black under `Exact`, mid-grey
/// otherwise. (A future enhancement could swap diagonals for a self-render
/// of the expert; v1 keeps the source-kind surface untouched.)
///
/// `input` is a local path or `hf://` URL. Repo-level `hf://` URLs are listed
/// over the HF API. When `stream` is false (the default), every shard is
/// downloaded to the local hf-hub cache and mmapped before any source is
/// constructed — per-cell diffs then read from local memory. When `stream`
/// is true, shards stay as `Data::Xet`/`Data::Http` and every per-tile
/// `fetch_range` issues an HTTP request. Single-file and local inputs always
/// go through `hf_url::resolve` + mmap regardless of the flag.
///
/// GGUF checkpoints with fused expert tensors (`ffn_*_exps.weight`) are out
/// of scope for v1 and return a clear error — see
/// `src/format/name_map.rs:61-69`.
pub async fn prepare_moe_diff_sources(
    input: &str,
    metric: format::DiffMetric,
    stream: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Open the input as one or more whole-file `Arc<Data>`s plus their formats.
    // Two code paths mirror `prepare_diff_sources` / `_from_http`: repo-level
    // `hf://` URLs stream lazily over HTTP; everything else is opened on disk
    // (downloading first if a single-file `hf://` URL was passed).
    let (datas, fmts, file_names) = if hf_url::is_repo_level(input)? {
        let listed = hf_url::list_repo_as_http_specs(input)
            .await
            .with_context(|| format!("listing files in {input}"))?;
        // Filter to model-format files only — config.json / tokenizer.* are
        // not relevant to per-expert layout.
        let st_specs: Vec<&(String, hf_url::RemoteFileSpec)> = listed
            .iter()
            .filter(|(n, _)| SourceFormat::from_name(n).is_some())
            .collect();
        if st_specs.is_empty() {
            anyhow::bail!("--moe-diff: no .safetensors / .gguf files in {input}");
        }
        let fmts: Vec<SourceFormat> = st_specs
            .iter()
            .map(|(n, _)| SourceFormat::from_name(n).unwrap_or(SourceFormat::Safetensors))
            .collect();
        let names: Vec<String> = st_specs.iter().map(|(n, _)| n.clone()).collect();

        let specs_owned: Vec<RemoteFileSpec> = st_specs.iter().map(|(_, s)| s.clone()).collect();
        // In streaming mode each shard becomes an `Arc<Data::Xet>` (or
        // `Data::Http` fallback) and per-tile diffs go over HTTP. In the
        // disk-backed default we skip the xet reconstruction setup entirely
        // — `materialize_remote_arcs` below downloads every shard via
        // hf-hub and replaces the arc with a `Data::Mapped` mmap, so the
        // per-tile path is pure memcpy.
        let mut datas: Vec<Arc<Data>> = if stream {
            let pb = setup_progress(
                "source files (xet reconstruction for moe-diff)",
                specs_owned.len() as u64,
            );
            let pb_for_workers = pb.clone();
            let mut out: Vec<(usize, anyhow::Result<Arc<Data>>)> = stream::iter(
                specs_owned.iter().cloned().enumerate(),
            )
            .map(|(i, spec)| {
                let pb = pb_for_workers.clone();
                async move {
                    let r: anyhow::Result<Arc<Data>> = if spec.xet_hash.is_some() {
                        match XetReader::new(&spec).await {
                            Ok(reader) => Ok(Arc::new(Data::Xet(reader))),
                            Err(e) => {
                                log::warn!(
                                    "{}: XetReader build failed ({e}); falling back to hf-hub Data::Http",
                                    spec.filename,
                                );
                                Ok(Arc::new(Data::Http {
                                    repo: spec.repo.clone(),
                                    filename: Arc::clone(&spec.filename),
                                    revision: Arc::clone(&spec.revision),
                                }))
                            }
                        }
                    } else {
                        Ok(Arc::new(Data::Http {
                            repo: spec.repo.clone(),
                            filename: Arc::clone(&spec.filename),
                            revision: Arc::clone(&spec.revision),
                        }))
                    };
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (i, r)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;
            if let Some(pb) = pb.as_ref() {
                pb.finish_and_clear();
            }
            out.sort_by_key(|(i, _)| *i);
            out.into_iter()
                .map(|(_, r)| r)
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            // Placeholder arcs that materialize_remote_arcs will replace.
            // Cheap (just an Arc<RemoteRepo> clone per shard); the real
            // download happens once in the helper.
            specs_owned
                .iter()
                .map(|spec| {
                    Arc::new(Data::Http {
                        repo: spec.repo.clone(),
                        filename: Arc::clone(&spec.filename),
                        revision: Arc::clone(&spec.revision),
                    })
                })
                .collect()
        };
        if !stream {
            materialize_remote_arcs(
                &mut datas,
                &specs_owned,
                "source files (downloading for moe-diff)",
            )
            .await?;
        }
        (datas, fmts, names)
    } else {
        // Local path or single-file hf:// URL. Resolve to disk and open.
        let resolved = hf_url::resolve(Path::new(input))
            .await
            .with_context(|| format!("resolving {input}"))?;
        let files: Vec<PathBuf> = if resolved.is_dir() {
            collect_files_recursive(&resolved)
                .into_iter()
                .filter(|p| SourceFormat::from_path(p).is_some())
                .collect()
        } else {
            vec![resolved]
        };
        if files.is_empty() {
            anyhow::bail!(
                "--moe-diff: no recognised model files (.safetensors / .gguf) in {input}"
            );
        }
        let mut datas = Vec::with_capacity(files.len());
        let mut fmts = Vec::with_capacity(files.len());
        let mut names = Vec::with_capacity(files.len());
        for p in &files {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            datas.push(Arc::new(Data::Mapped(unsafe { Mmap::map(&f) }?)));
            fmts.push(SourceFormat::from_path(p).unwrap_or(SourceFormat::Safetensors));
            names.push(
                p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        (datas, fmts, names)
    };

    // Fetch headers for every shard in parallel — same pattern as
    // `build_multi_safetensors_diff_sources_inner`.
    let total_headers = datas.len() as u64;
    let pb = setup_progress("source files (model headers)", total_headers);
    let headers: Vec<(usize, Vec<format::TensorMeta>)> = {
        let pb = pb.clone();
        // Materialize the iter input as owned tuples so the
        // `stream::iter(...).map(...)` closure doesn't borrow `datas` /
        // `fmts` across the await — the for-all-lifetimes Send bound on
        // the `MoeDiffPrep` trait future doesn't tolerate borrows
        // leaking through async_trait's erased future.
        let items: Vec<(usize, Arc<Data>, SourceFormat)> = datas
            .iter()
            .zip(fmts.iter())
            .enumerate()
            .map(|(i, (d, fmt))| (i, Arc::clone(d), *fmt))
            .collect();
        let mut out: Vec<(usize, anyhow::Result<Vec<format::TensorMeta>>)> = stream::iter(items)
            .map(|(i, d, fmt)| {
                let pb = pb.clone();
                async move {
                    let r = fetch_model_header(&d, fmt)
                        .await
                        .map(|(t, _)| t)
                        .with_context(|| format!("reading {fmt:?} header for moe-diff file {i}"));
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (i, r)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;
        out.sort_by_key(|(i, _)| *i);
        out.into_iter()
            .map(|(i, r)| r.map(|t| (i, t)))
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    // Detect GGUF fused-experts up front and bail with a clear message.
    for (shard_idx, tensors) in &headers {
        if matches!(fmts[*shard_idx], SourceFormat::Gguf)
            && tensors
                .iter()
                .any(|t| crate::format::moe::is_fused_gguf_expert(&t.name))
        {
            anyhow::bail!(
                "--moe-diff: GGUF fused expert tensors are not yet supported \
                 (found `ffn_*_exps.weight` in {}). See src/format/name_map.rs:61-69 — \
                 the canonicaliser would need to either preserve per-expert names or \
                 expose a slice-by-expert path.",
                file_names
                    .get(*shard_idx)
                    .map(String::as_str)
                    .unwrap_or("<unknown>"),
            );
        }
    }

    // Group per-expert tensors by (layer, weight, expert_idx). Each entry
    // carries the shard index + the tensor metadata.
    use crate::format::moe::{parse_hf_expert, ExpertWeight as EW};
    use std::collections::BTreeMap;
    type LayerKey = (u32, EW);
    let mut groups: BTreeMap<LayerKey, BTreeMap<u32, (usize, format::TensorMeta)>> =
        BTreeMap::new();
    let mut shapes_seen: BTreeMap<LayerKey, Vec<u64>> = BTreeMap::new();
    let mut dtypes_seen: BTreeMap<LayerKey, format::Dtype> = BTreeMap::new();
    for (shard_idx, tensors) in headers {
        for t in tensors {
            if let Some(r) = parse_hf_expert(&t.name) {
                let key = (r.layer_idx, r.weight);
                // Sanity: all experts of one (layer, weight) must share shape
                // + dtype (they do in practice — they're the same per-expert
                // FFN slice). Mismatch would indicate a malformed checkpoint;
                // log loudly and skip the offending tensor.
                if let Some(prev_shape) = shapes_seen.get(&key) {
                    if prev_shape != &t.shape {
                        log::warn!(
                            "--moe-diff: shape mismatch within layer {} weight {} expert {}: \
                             {:?} vs {:?} — skipping",
                            r.layer_idx,
                            r.weight.label(),
                            r.expert_idx,
                            prev_shape,
                            &t.shape,
                        );
                        continue;
                    }
                } else {
                    shapes_seen.insert(key, t.shape.clone());
                    dtypes_seen.insert(key, t.dtype);
                }
                groups
                    .entry(key)
                    .or_default()
                    .insert(r.expert_idx, (shard_idx, t));
            }
        }
    }

    if groups.is_empty() {
        anyhow::bail!(
            "--moe-diff: no per-expert tensors found in {input} \
             (expected `model.layers.{{L}}.mlp.experts.{{E}}.{{gate|up|down}}_proj.weight`)"
        );
    }

    // For DiffMetric::Rms we need a per-(layer, weight) RMS estimate. Sample
    // from expert 0's bytes — every expert of one layer+weight shares shape
    // and lives in the same shard family, so one sample per group is enough.
    let need_rms = matches!(metric, format::DiffMetric::Rms);
    let mut scales: BTreeMap<LayerKey, f32> = BTreeMap::new();
    if need_rms {
        const SCALE_SAMPLE_BYTES: u64 = 64 * 1024;
        let pb = setup_progress("moe expert RMS samples", groups.len() as u64);
        type SampleJob = (LayerKey, usize, u64, u64, format::Dtype);
        let inputs: Vec<SampleJob> = groups
            .iter()
            .filter_map(|(k, experts)| {
                let (shard_idx, meta) = experts.values().next()?;
                let elem = meta.dtype.element_size() as u64;
                let tensor_bytes = meta.file_end.saturating_sub(meta.file_start);
                let want = SCALE_SAMPLE_BYTES.min(tensor_bytes);
                let len = want.checked_div(elem).map(|q| q * elem).unwrap_or(0);
                Some((*k, *shard_idx, meta.file_start, len, meta.dtype))
            })
            .collect();
        let datas_ref = &datas;
        let pb_for_workers = pb.clone();
        let results: Vec<(LayerKey, f32)> = stream::iter(inputs)
            .map(|(k, shard, start, len, dtype)| {
                let d = Arc::clone(&datas_ref[shard]);
                let pb = pb_for_workers.clone();
                async move {
                    let v = if len == 0 {
                        0.0
                    } else {
                        match d.fetch_range(start, len as usize).await {
                            Ok(bytes) => format::rms_from_buf(dtype, &bytes),
                            Err(e) => {
                                log::warn!("moe-diff: RMS sample failed ({e}); using 0.0");
                                0.0
                            }
                        }
                    };
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (k, v)
                }
            })
            .buffer_unordered(RMS_SAMPLE_FETCH_CONCURRENCY)
            .collect()
            .await;
        if let Some(pb) = pb.as_ref() {
            pb.finish_and_clear();
        }
        for (k, v) in results {
            scales.insert(k, v);
        }
    }

    // Emit one Source per upper-triangle cell (including diagonal). The
    // diff Source carries both `orig` and `mod_` pointing at the same
    // checkpoint's `Arc<Data>`, just at different file_start offsets — the
    // existing TensorDiff render path handles the rest.
    let mut sources: Vec<Source> = Vec::new();
    let mut total: u64 = 0;
    for ((layer_idx, weight), experts) in &groups {
        let scale_orig = scales.get(&(*layer_idx, *weight)).copied().unwrap_or(0.0);
        // Stable expert order ascending.
        let expert_ids: Vec<u32> = experts.keys().copied().collect();
        for (a, &ei) in expert_ids.iter().enumerate() {
            for &ej in &expert_ids[a..] {
                let (oi, e_i) = &experts[&ei];
                let (oj, e_j) = &experts[&ej];
                let nelem: u64 = e_i.shape.iter().product();
                if nelem == 0 {
                    continue;
                }
                // Synthetic 1-byte-per-element diff tensor that the
                // architectural layout places at its natural 2D shape.
                let diff_meta = format::TensorMeta {
                    name: format!(
                        "moe::L{layer}::{weight}::E{i}-E{j}",
                        layer = layer_idx,
                        weight = weight.label(),
                        i = ei,
                        j = ej,
                    ),
                    dtype: format::Dtype::U8,
                    shape: e_i.shape.clone(),
                    file_start: 0,
                    file_end: nelem,
                    packed_sidecars: None,
                };
                let label = if ei == ej {
                    format!(
                        "Layer {layer_idx} · {wlabel} · Expert {ei} (self)",
                        wlabel = weight.label()
                    )
                } else {
                    format!(
                        "Layer {layer_idx} · {wlabel} · Expert {ei} × Expert {ej}",
                        wlabel = weight.label()
                    )
                };
                let mut extensions = Extensions::default();
                extensions.insert(ModelInfo {
                    format: SourceFormat::Safetensors,
                    tensors: vec![diff_meta],
                    color_ranges: Vec::new(),
                });
                extensions.insert(MoeCell {
                    layer: *layer_idx,
                    weight: *weight,
                    i: ei,
                    j: ej,
                });
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Custom(Box::new(TensorDiffSource {
                        orig: Arc::clone(&datas[*oi]),
                        mod_: Arc::clone(&datas[*oj]),
                        orig_start: e_i.file_start,
                        mod_start: e_j.file_start,
                        orig_dtype: e_i.dtype,
                        mod_dtype: e_j.dtype,
                        metric,
                        scale_orig,
                        byte_size: nelem,
                    })),
                    byte_size: nelem,
                    name_override: Some(label),
                    xet_terms: None,
                    extensions,
                });
                total += nelem;
            }
        }
    }

    let n_layers = groups
        .keys()
        .map(|k| k.0)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let weight_keys = groups
        .keys()
        .filter(|(_, w)| matches!(w, EW::GateProj))
        .count();
    log::info!(
        "moe-diff: {} layer(s), {} weight slot(s) per layer ({}{} {}), \
         emitted {} cell(s) totalling {} synthetic byte(s)",
        n_layers,
        groups.len() / n_layers.max(1),
        if groups.contains_key(&(groups.keys().next().unwrap().0, EW::GateProj)) {
            "gate_proj "
        } else {
            ""
        },
        if groups.contains_key(&(groups.keys().next().unwrap().0, EW::UpProj)) {
            "up_proj "
        } else {
            ""
        },
        if groups.contains_key(&(groups.keys().next().unwrap().0, EW::DownProj)) {
            "down_proj"
        } else {
            ""
        },
        sources.len(),
        total,
    );
    let _ = weight_keys; // intentional: silence unused if log changes

    Ok((sources, total))
}

/// Build per-tensor diff Sources from two single .safetensors files.
pub async fn build_safetensors_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    metric: format::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    build_multi_safetensors_diff_sources(
        &[original.to_path_buf()],
        &[modified.to_path_buf()],
        is_finetune,
        metric,
    )
    .await
}

/// Sidecar metadata fetched alongside a safetensors source. Both fields are
/// optional — the architectural layout works without them by inferring
/// everything from tensor names + shapes; when present they let the layout
/// validate inferred layer counts and reserve canonical slots for tensors
/// that live in shards we didn't load.
#[derive(Default, Debug, Clone)]
pub struct SourceMeta {
    pub config: Option<ModelConfig>,
    pub index: Option<SafetensorsIndex>,
}

/// Best-effort opportunistic load of `config.json` and
/// `model.safetensors.index.json` for a single source. Local sources read
/// their parent directory; HF Hub sources fetch the file from the same repo;
/// other kinds return empty.
///
/// Errors are swallowed and logged at debug level — sidecar info is advisory,
/// and a missing sidecar must not break rendering.
// `try_load_source_meta` / `load_meta_for_sources` / `fetch_hf_sidecar`
// are the sidecar-fetching cluster: they load `config.json` and
// `model.safetensors.index.json` next to a source so the arch layout
// can read transformer hyperparams (num_hidden_layers, etc.) and shard
// stitching info. They're invoked by [`crate::SourceMetaSidecarHook`],
// the [`arbvis::PrepareSourcesExtension`] impl registered on the
// registry by [`crate::register_all`]. The hook runs once at the top
// of `dispatch_render`, after every `Source` has been built, and
// inserts a `SourceMeta` into each source's extensions for
// `ArchLayoutPlugin::build` to read back.
pub async fn try_load_source_meta(source: &Source) -> SourceMeta {
    match &source.kind {
        SourceKind::File(p) => {
            // For local GGUF files, the equivalent of `config.json` is
            // embedded in the binary header. Re-parse the prefix to extract
            // it; this is cheap (~1 MiB read at setup time only).
            if matches!(SourceFormat::from_path(p), Some(SourceFormat::Gguf)) {
                if let Ok(meta_size) = std::fs::metadata(p).map(|m| m.len()) {
                    let want = GGUF_HEADER_FETCH_INITIAL.min(meta_size as usize);
                    if let Ok(mut f) = File::open(p) {
                        let mut buf = vec![0u8; want];
                        if f.read_exact(&mut buf).is_ok() {
                            if let Ok(h) = format::gguf::parse_header(&buf) {
                                return SourceMeta {
                                    config: Some(ModelConfig::from_gguf_metadata(&h.metadata)),
                                    index: None,
                                };
                            }
                        }
                    }
                }
            }
            if let Some(dir) = p.parent() {
                SourceMeta {
                    config: ModelConfig::try_from_dir(dir),
                    index: SafetensorsIndex::try_from_dir(dir),
                }
            } else {
                SourceMeta::default()
            }
        }
        SourceKind::Http(spec) => {
            // For remote GGUF files, fetch the header prefix and pull
            // ModelConfig fields out of its KV table instead of looking for
            // a sibling config.json (which often doesn't exist for
            // llama.cpp-style GGUF releases).
            if matches!(
                SourceFormat::from_name(&spec.filename),
                Some(SourceFormat::Gguf)
            ) {
                let data = Data::Http {
                    repo: spec.repo.clone(),
                    filename: Arc::clone(&spec.filename),
                    revision: Arc::clone(&spec.revision),
                };
                if let Ok(buf) = data.fetch_range(0, GGUF_HEADER_FETCH_INITIAL).await {
                    if let Ok(h) = format::gguf::parse_header(&buf) {
                        return SourceMeta {
                            config: Some(ModelConfig::from_gguf_metadata(&h.metadata)),
                            index: None,
                        };
                    }
                }
                return SourceMeta::default();
            }
            // Fetch config.json and model.safetensors.index.json from the
            // same repo via a raw GET. Both files are small enough that we
            // don't bother streaming.
            let config = fetch_hf_sidecar(&spec.repo, &spec.revision, "config.json")
                .await
                .ok()
                .and_then(|b| ModelConfig::from_bytes(&b));
            let index =
                fetch_hf_sidecar(&spec.repo, &spec.revision, "model.safetensors.index.json")
                    .await
                    .ok()
                    .and_then(|b| SafetensorsIndex::from_bytes(&b));
            SourceMeta { config, index }
        }
        _ => SourceMeta::default(),
    }
}

/// Load sidecar meta for every source, de-duplicated by HF Hub repo+revision
/// (so a 4-shard model triggers one config.json fetch, not four). Returns a
/// `Vec` parallel to `sources`.
pub async fn load_meta_for_sources(sources: &[Source]) -> Vec<SourceMeta> {
    // Group sources by (repo_id, revision) for HF Hub; by parent dir for
    // local. Each group gets one fetch.
    let mut keys: Vec<Option<String>> = Vec::with_capacity(sources.len());
    for s in sources {
        let key = match &s.kind {
            SourceKind::File(p) => p.parent().map(|d| format!("local:{}", d.display())),
            SourceKind::Http(spec) => Some(format!("hf:{}@{}", spec.repo.repo_id(), spec.revision)),
            _ => None,
        };
        keys.push(key);
    }
    let mut cache: HashMap<String, SourceMeta> = HashMap::new();
    let mut result: Vec<SourceMeta> = Vec::with_capacity(sources.len());
    for (s, k) in sources.iter().zip(keys.iter()) {
        match k {
            Some(k) => {
                if let Some(m) = cache.get(k) {
                    result.push(m.clone());
                } else {
                    let m = try_load_source_meta(s).await;
                    cache.insert(k.clone(), m.clone());
                    result.push(m);
                }
            }
            None => result.push(SourceMeta::default()),
        }
    }
    result
}

/// Fetch a small sidecar file (e.g. `config.json`) from an HF Hub repo via
/// a raw GET to `{endpoint}/{repo_id}/resolve/{revision}/{filename}`.
/// Returns the raw bytes on success.
async fn fetch_hf_sidecar(
    repo: &RemoteRepo,
    revision: &str,
    filename: &str,
) -> anyhow::Result<Vec<u8>> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        hf_url::endpoint(),
        repo.repo_id(),
        revision,
        filename,
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .with_context(|| format!("building reqwest client for {filename}"))?;
    let mut req = client.get(&url);
    if let Some(tok) = hf_url::read_token() {
        req = req.bearer_auth(tok);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("fetching {url}"))?;
    let resp = resp
        .error_for_status()
        .with_context(|| format!("non-2xx for {filename}"))?;
    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("reading body of {filename}"))?;
    Ok(bytes.to_vec())
}
