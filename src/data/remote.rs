//! Remote (HF Hub) source plumbing: `prepare_sources_from_specs` (mixed
//! local/remote input specs), and the download/materialization helpers
//! (`materialize_http_sources`, `download_specs_to_paths`, `populate_xet_terms`)
//! shared by the render entry points and the diff builders.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::hf_url::RemoteFileSpec;
use crate::xet::{self, XetTerm};
use futures::stream::{self, StreamExt};

use super::source::{collect_files_recursive, setup_progress};
use super::{Data, Extensions, InputSpec, Source, SourceKind};

/// Bounded concurrency for setup-time HTTP loops (xet reconstruction,
/// safetensors header fetches, non-safetensors diff downloads). The global
/// AIMD throttle still caps the *actual* in-flight count; this just lets the
/// runtime have enough simultaneous awaiting tasks to keep the throttle full.
const SETUP_FETCH_CONCURRENCY: usize = 16;

/// Build sources from a mixed list of local paths and remote HF file specs.
/// Remote specs are turned into `SourceKind::Http` entries (no download).
///
/// Async so the remote arm can call each [`crate::registry::FormatPlugin`]'s
/// `populate_remote` over a `Data::Http` handle — that's the only way
/// `ModelInfo` gets stuffed into `Source.extensions` when `--stream` /
/// `--show-xet-xorbs` keep the file remote. Without that population, the
/// arch layout's `applicable()` check returns false on every remote source
/// and arbvis silently falls back to byte-Hilbert.
pub async fn prepare_sources_from_specs(
    specs: &[InputSpec],
    registry: &crate::registry::Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
    if specs.is_empty() {
        log::info!("Reading stdin...");
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                kind: SourceKind::Buffered(buf),
                byte_size: len,
                name_override: None,
                xet_terms: None,
                extensions: Extensions::default(),
            }],
            len,
        ));
    }

    // Pre-expand `InputSpec::Local(dir)` into one `InputSpec::Local(file)` per
    // contained file, matching the behaviour of [`prepare_sources`]. Without
    // this, a local directory pushed into the specs (e.g. by `--stream
    // ./snapshots/llama-7b/`) would yield a single Source pointing at the
    // directory itself — load_source_data then mmaps the dir and fails.
    let expanded: Vec<InputSpec> = specs
        .iter()
        .flat_map(|spec| match spec {
            InputSpec::Local(p) if p.is_dir() => collect_files_recursive(p)
                .into_iter()
                .map(InputSpec::Local)
                .collect::<Vec<_>>(),
            InputSpec::Local(p) => vec![InputSpec::Local(p.clone())],
            InputSpec::Remote(s) => vec![InputSpec::Remote(s.clone())],
        })
        .collect();

    let mut sources = Vec::new();
    let mut total = 0u64;

    for spec in &expanded {
        match spec {
            InputSpec::Local(path) => {
                let size = match std::fs::metadata(path) {
                    Ok(m) => m.len(),
                    Err(e) => {
                        log::warn!("{}: {} — skipping", path.display(), e);
                        continue;
                    }
                };
                let mut extensions = Extensions::default();
                for plugin in &registry.formats {
                    if plugin.detects_path(path.as_path()) {
                        if let Err(e) = plugin.populate_local(path.as_path(), size, &mut extensions)
                        {
                            log::warn!(
                                "{}: format plugin `{}` failed: {e} — treating as plain binary",
                                path.display(),
                                plugin.id()
                            );
                        }
                        break;
                    }
                }
                total += size;
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::File(path.clone()),
                    byte_size: size,
                    name_override: None,
                    xet_terms: None,
                    extensions,
                });
            }
            InputSpec::Remote(spec) => {
                let size = spec.size;
                total += size;
                // Open a `Data::Http` handle pointing at the same remote file
                // the downstream loader/renderer will use. This costs nothing
                // up front — `Data::Http` is just (repo, filename, revision);
                // `populate_remote` is what issues the actual head-prefix
                // range fetch needed to parse the format header.
                let data = Data::Http {
                    repo: spec.repo.clone(),
                    filename: Arc::clone(&spec.filename),
                    revision: Arc::clone(&spec.revision),
                };
                let mut extensions = Extensions::default();
                let filename_path = Path::new(spec.filename.as_str());
                for plugin in &registry.formats {
                    if plugin.detects_path(filename_path) {
                        if let Err(e) = plugin.populate_remote(&data, size, &mut extensions).await {
                            // Non-fatal: arch layout falls back to byte-Hilbert,
                            // the same way it would for any source whose format
                            // plugin couldn't parse its header.
                            log::warn!(
                                "{}: format plugin `{}` (remote) failed: {e} — \
                                 treating as plain binary",
                                crate::hf_url::sanitize_log_text(spec.filename.as_str()),
                                plugin.id()
                            );
                        }
                        break;
                    }
                }
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Http(spec.clone()),
                    byte_size: size,
                    name_override: None,
                    xet_terms: None,
                    extensions,
                });
            }
        }
    }

    Ok((sources, total))
}

/// Materialize every `SourceKind::Http` source as a local file via one
/// whole-file download per source, then swap each source to `SourceKind::File`.
///
/// Why: `Data::Http::fetch_range` issues a fresh HTTPS GET (with Range) per
/// tile. With tens of thousands of tiles per file, the per-call TLS + HTTP
/// setup dominates and the pipeline appears stalled.
///
/// One whole-file `hf download` per source amortises that overhead across
/// the entire file (which the renderer will read every byte of anyway). After
/// materialization, all tile reads are mmap'd `memcpy`s — no HTTP, no throttle.
///
/// `populate_xet_terms` must run *before* this so the xet term metadata is
/// captured from the still-remote `RemoteFileSpec`.
pub async fn materialize_http_sources(sources: &mut [Source]) -> anyhow::Result<()> {
    // Snapshot (index, spec) for Http sources so the futures don't borrow `sources`.
    let jobs: Vec<(usize, RemoteFileSpec)> = sources
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match &s.kind {
            SourceKind::Http(spec) => Some((i, spec.clone())),
            _ => None,
        })
        .collect();

    if jobs.is_empty() {
        return Ok(());
    }

    let indices: Vec<usize> = jobs.iter().map(|(i, _)| *i).collect();
    let specs: Vec<RemoteFileSpec> = jobs.into_iter().map(|(_, s)| s).collect();
    let paths = download_specs_to_paths(&specs, "source files (downloading for xet view)").await?;

    for (i, path) in indices.into_iter().zip(paths) {
        // Preserve display name + xet_terms; only the storage kind changes.
        let display = sources[i].name();
        sources[i].kind = SourceKind::File(path);
        if sources[i].name_override.is_none() {
            sources[i].name_override = Some(display);
        }
    }
    Ok(())
}

/// Download a batch of [`RemoteFileSpec`]s to the local HF cache and return
/// the local paths in the same order. Drives the AIMD throttle through
/// [`crate::throttle::with_throttle`] and reports progress via a one-shot
/// `setup_progress` bar.
///
/// Each download is a fresh `hf download <repo> <file>` subprocess. The CLI
/// applies its own `HF_HUB_DOWNLOAD_TIMEOUT` (default 10 s) to detect CDN
/// stalls, so we no longer need a custom watchdog around the call. Errors
/// classify via `HfCliError::ErrorClassify`, which feeds the throttle's
/// existing retry/backoff path.
///
/// Shared by every disk-backed materialisation path:
/// [`materialize_http_sources`] (normal flow's `SourceKind::Http` swap) and
/// `materialize_remote_arcs` (the `Arc<Data>`s buried inside
/// `SourceKind::TensorDiff` for `--diff`).
pub async fn download_specs_to_paths(
    specs: &[RemoteFileSpec],
    progress_label: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    use crate::hf_cli;
    use crate::throttle::with_throttle;

    let pb = setup_progress(progress_label, specs.len() as u64);
    let pb_for_workers = pb.clone();

    let mut downloads: Vec<(usize, anyhow::Result<PathBuf>)> =
        stream::iter(specs.iter().cloned().enumerate())
            .map(|(i, spec)| {
                let pb = pb_for_workers.clone();
                async move {
                    let filename = (*spec.filename).clone();
                    let revision = (*spec.revision).clone();
                    let label = format!(
                        "hf download {}",
                        crate::hf_url::sanitize_log_text(&filename),
                    );
                    let repo_id = spec.repo.repo_id().to_string();
                    // Map the api_segment ("models"/"datasets"/"spaces") to the
                    // `--type` flag value the CLI expects.
                    let repo_type = match spec.repo.api_segment() {
                        "models" => "model",
                        "datasets" => "dataset",
                        "spaces" => "space",
                        other => {
                            return (
                                i,
                                Err(anyhow::anyhow!(
                                    "unexpected api_segment {other:?} on remote repo"
                                )),
                            );
                        }
                    };
                    let result: Result<PathBuf, anyhow::Error> = with_throttle(&label, || async {
                        hf_cli::download([
                            "download",
                            "--type",
                            repo_type,
                            "--revision",
                            revision.as_str(),
                            repo_id.as_str(),
                            filename.as_str(),
                        ])
                        .await
                    })
                    .await
                    .map_err(anyhow::Error::from);
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (i, result)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;

    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    downloads.sort_by_key(|(i, _)| *i);
    downloads
        .into_iter()
        .map(|(_, r)| r)
        .collect::<anyhow::Result<Vec<_>>>()
}

/// Fetch xet reconstruction terms for any HTTP-backed sources.
///
/// Each source gets `xet_terms = Some(vec)` — empty for sources without a xet
/// hash (local files, non-xet remote files), populated for xet-backed remote
/// sources. Errors from the xet endpoints propagate up.
pub async fn populate_xet_terms(sources: &mut [Source]) -> anyhow::Result<()> {
    // Fetch xet reconstructions concurrently — each Http source needs two
    // throttled HTTP round-trips (`xet-read-token` + `reconstructions/{hash}`)
    // and the global throttle caps the real concurrency, so let the runtime
    // have plenty of awaiting tasks.
    let pb = setup_progress("source files (xet reconstruction)", sources.len() as u64);
    let pb_for_close = pb.clone();

    // Decouple the per-source future from the borrow on `sources` by snapshotting
    // the index + http spec, then writing results back by index.
    let jobs: Vec<(usize, Option<RemoteFileSpec>)> = sources
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let spec = if let SourceKind::Http(spec) = &s.kind {
                Some(spec.clone())
            } else {
                None
            };
            (i, spec)
        })
        .collect();

    let pb_for_workers = pb.clone();
    let mut results: Vec<(usize, anyhow::Result<Vec<XetTerm>>)> = stream::iter(jobs)
        .map(|(i, maybe_spec)| {
            let pb = pb_for_workers.clone();
            async move {
                let terms = match maybe_spec {
                    Some(spec) => xet::reconstruction_for(&spec).await,
                    None => Ok(Vec::new()),
                };
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                (i, terms)
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;

    if let Some(pb) = pb_for_close.as_ref() {
        pb.finish_and_clear();
    }

    // Stable order: write each result back to its original index. The first
    // error wins; later sources still get `xet_terms = None` so the caller
    // can distinguish "didn't fetch" from "fetched but empty".
    results.sort_by_key(|(i, _)| *i);
    for (i, r) in results {
        sources[i].xet_terms = Some(r?);
    }
    Ok(())
}
