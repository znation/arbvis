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
/// Also shared with `providers::resolve_input_sources` (input resolution), so
/// every setup-time stage stays user-visibly consistent.
pub(crate) const SETUP_FETCH_CONCURRENCY: usize = 16;

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
    // directory itself — load_source_data then reads the dir and fails.
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

    // Remote header population is an independent network round-trip per source
    // (each plugin's `populate_remote` issues a head-prefix range fetch). Run
    // those fetches concurrently instead of one-at-a-time: a repo-level
    // `hf://` URL expands to one spec per Hub tree entry, so serializing here
    // puts a whole round-trip on the critical path per file. Results are
    // reattached by position so source order (and thus labels and byte
    // offsets) stays deterministic.
    let mut remote_exts: Vec<Extensions> =
        (0..expanded.len()).map(|_| Extensions::default()).collect();
    let remote_slots: Vec<usize> = expanded
        .iter()
        .enumerate()
        .filter_map(|(i, spec)| match spec {
            InputSpec::Remote(_) => Some(i),
            _ => None,
        })
        .collect();
    let fetched: Vec<Extensions> = stream::iter(remote_slots.iter().copied())
        .map(|i| {
            let spec = match &expanded[i] {
                InputSpec::Remote(s) => s.clone(),
                _ => unreachable!("remote_slots holds only Remote indices"),
            };
            async move {
                let data = Data::Http {
                    repo: spec.repo.clone(),
                    filename: Arc::clone(&spec.filename),
                    revision: Arc::clone(&spec.revision),
                };
                let extensions = registry
                    .populate_remote_extensions(
                        Path::new(spec.filename.as_str()),
                        &data,
                        spec.size,
                        &crate::hf_url::sanitize_log_text(spec.filename.as_str()),
                    )
                    .await;
                extensions
            }
        })
        .buffered(SETUP_FETCH_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;
    for (slot, extensions) in remote_slots.iter().copied().zip(fetched) {
        remote_exts[slot] = extensions;
    }

    let mut sources = Vec::new();
    let mut total = 0u64;

    let mut next_remote = 0usize;
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
                let extensions = registry.populate_local_extensions(
                    path.as_path(),
                    size,
                    &path.display().to_string(),
                );
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
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Http(spec.clone()),
                    byte_size: size,
                    name_override: None,
                    xet_terms: None,
                    // Populated by the concurrent pass above; take it so the
                    // Extensions map moves instead of cloning.
                    extensions: std::mem::take(&mut remote_exts[remote_slots[next_remote]]),
                });
                next_remote += 1;
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
/// materialization, all tile reads are memcpys off in-memory snapshots — no
/// HTTP, no throttle.
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
/// `crate::throttle::with_throttle` and reports progress via a one-shot
/// `setup_progress` bar.
///
/// Each download is a fresh `hf download <repo> <file>` subprocess. The CLI
/// applies its own `HF_HUB_DOWNLOAD_TIMEOUT` (default 10 s) to detect CDN
/// stalls, so we no longer need a custom watchdog around the call. Errors
/// classify via `HfCliError::ErrorClassify`, which feeds the throttle's
/// existing retry/backoff path.
///
/// Shared by every disk-backed materialisation path:
/// [`materialize_http_sources`] (normal flow's `SourceKind::Http` swap) and,
/// downstream, any plugin that materialises remote sources to local paths
/// before rendering.
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
                    // A repo-level `hf://` URL expands to per-file specs whose
                    // filenames come from the Hub tree listing, so a hostile
                    // repo controls `filename` here. Passed positionally, a
                    // name beginning with `-` would be parsed by the `hf` CLI
                    // as an option instead of a path (argument injection).
                    // Such a name already misparses today, so rejecting it
                    // with a clear error is a strict improvement.
                    if filename.starts_with('-') {
                        return (
                            i,
                            Err(anyhow::anyhow!(
                                "refusing `hf download` for {:?}: filename starts with `-` \
                                 and would be parsed as a CLI flag",
                                crate::hf_url::sanitize_log_text(&filename),
                            )),
                        );
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_url::RepoKind;
    use crate::registry::{FormatPlugin, Registry};
    use crate::test_plugin::{FakePlugin, Tag};
    use futures::future::BoxFuture;
    use std::fs;
    use tempfile::TempDir;

    fn local_spec(p: &Path) -> InputSpec {
        InputSpec::Local(p.to_path_buf())
    }

    fn remote_spec(repo: &crate::hf_url::RemoteRepo, filename: &str, size: u64) -> InputSpec {
        InputSpec::Remote(crate::hf_url::RemoteFileSpec {
            repo: repo.clone(),
            filename: Arc::new(filename.to_string()),
            revision: Arc::new("main".to_string()),
            size,
            xet_hash: None,
        })
    }

    /// Format plugin whose `populate_remote` tags the extensions map and —
    /// while running — tracks the maximum number of concurrently in-flight
    /// `populate_remote` calls across all specs, so tests can prove the
    /// remote header pass actually overlaps instead of serializing.
    struct ProbePlugin {
        ext: &'static str,
        state: Arc<ProbeState>,
    }

    struct ProbeState {
        in_flight: std::sync::atomic::AtomicUsize,
        max_in_flight: std::sync::atomic::AtomicUsize,
    }

    impl ProbePlugin {
        fn new(ext: &'static str) -> (Self, Arc<ProbeState>) {
            let state = Arc::new(ProbeState {
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                max_in_flight: std::sync::atomic::AtomicUsize::new(0),
            });
            (
                Self {
                    ext,
                    state: Arc::clone(&state),
                },
                state,
            )
        }
    }

    impl FormatPlugin for ProbePlugin {
        fn id(&self) -> &'static str {
            "probe"
        }
        fn detects_path(&self, path: &Path) -> bool {
            path.extension().and_then(|e| e.to_str()) == Some(self.ext)
        }
        fn populate_local(
            &self,
            _path: &Path,
            _file_size: u64,
            exts: &mut Extensions,
        ) -> anyhow::Result<()> {
            exts.insert(Tag("local".to_string()));
            Ok(())
        }
        fn populate_remote<'a>(
            &'a self,
            _data: &'a Data,
            _byte_size: u64,
            exts: &'a mut Extensions,
        ) -> BoxFuture<'a, anyhow::Result<()>> {
            Box::pin(async move {
                let now = self
                    .state
                    .in_flight
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    + 1;
                self.state
                    .max_in_flight
                    .fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                // Hold the slot long enough that a serial pass could not
                // overlap consecutive calls.
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                self.state
                    .in_flight
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                exts.insert(Tag("remote".to_string()));
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn dir_spec_expands_to_one_source_per_file_recursively() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("b.bin"), b"12345").unwrap();
        fs::create_dir(tmp.path().join("sub")).unwrap();
        fs::write(tmp.path().join("sub").join("a.txt"), b"123").unwrap();
        fs::create_dir(tmp.path().join("sub").join("deep")).unwrap();
        fs::write(tmp.path().join("sub").join("deep").join("c"), b"1").unwrap();

        let (sources, total) =
            prepare_sources_from_specs(&[local_spec(tmp.path())], &Registry::default())
                .await
                .unwrap();

        // collect_files_recursive sorts full paths, so b.bin < sub/a.txt <
        // sub/deep/c at this temp path.
        let names: Vec<String> = sources
            .iter()
            .map(|s| match &s.kind {
                SourceKind::File(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
                _ => panic!("expected File source"),
            })
            .collect();
        assert_eq!(names, vec!["b.bin", "a.txt", "c"]);
        assert_eq!(total, 3 + 5 + 1);
        for (i, s) in sources.iter().enumerate() {
            assert_eq!(s.file_idx, i);
            assert_eq!(
                s.byte_size,
                match i {
                    0 => 5,
                    1 => 3,
                    _ => 1,
                }
            );
            assert!(matches!(s.kind, SourceKind::File(_)));
            assert!(s.name_override.is_none());
            assert!(s.xet_terms.is_none());
        }
    }

    #[tokio::test]
    async fn missing_local_file_is_skipped_not_fatal() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nope.bin");
        fs::write(tmp.path().join("real.bin"), b"xy").unwrap();

        let (sources, total) = prepare_sources_from_specs(
            &[
                local_spec(&missing),
                local_spec(&tmp.path().join("real.bin")),
            ],
            &Registry::default(),
        )
        .await
        .unwrap();

        // The missing spec is dropped with a warning; the real file keeps the
        // only slot and file_idx stays contiguous (0).
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].byte_size, 2);
        assert_eq!(sources[0].file_idx, 0);
        assert_eq!(total, 2);
    }

    #[tokio::test]
    async fn first_matching_format_plugin_wins_and_fills_extensions() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("model.foo");
        fs::write(&f, b"abc").unwrap();

        let registry = Registry {
            formats: vec![
                Arc::new(FakePlugin {
                    ext: "foo",
                    tag: "first",
                    fail: false,
                }),
                Arc::new(FakePlugin {
                    ext: "foo",
                    tag: "second",
                    fail: false,
                }),
            ],
            ..Registry::default()
        };

        let (sources, _) = prepare_sources_from_specs(&[local_spec(&f)], &registry)
            .await
            .unwrap();

        assert_eq!(sources.len(), 1);
        let tag = sources[0].extensions.get::<Tag>().unwrap();
        assert_eq!(tag.0, "first");
    }

    #[tokio::test]
    async fn failing_format_plugin_is_non_fatal_plain_binary() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("broken.foo");
        fs::write(&f, b"abc").unwrap();

        let registry = Registry {
            formats: vec![Arc::new(FakePlugin {
                ext: "foo",
                tag: "never",
                fail: true,
            })],
            ..Registry::default()
        };

        let (sources, total) = prepare_sources_from_specs(&[local_spec(&f)], &registry)
            .await
            .unwrap();

        // The failure degrades to plain binary: source still produced, no tag
        // attached, size still counted.
        assert_eq!(sources.len(), 1);
        assert!(sources[0].extensions.get::<Tag>().is_none());
        assert_eq!(total, 3);
    }

    /// The remote header pass must run concurrently: with several remote
    /// specs, `populate_remote` calls overlap (max in-flight > 1), instead of
    /// serializing one network round-trip per file. 4 specs × 30 ms hold ≈
    /// 120 ms overlapped vs ≈ 120 ms serialized — the probe sees the
    /// difference deterministically.
    #[tokio::test]
    async fn remote_header_population_overlaps_across_specs() {
        let repo = crate::hf_url::remote_repo_for_tests(RepoKind::Model, "overlap/repo");
        let specs: Vec<InputSpec> = (0..4)
            .map(|i| remote_spec(&repo, &format!("f{i}.foo"), 1))
            .collect();
        let (plugin, state) = ProbePlugin::new("foo");
        let registry = Registry {
            formats: vec![Arc::new(plugin)],
            ..Registry::default()
        };

        let (sources, total) = prepare_sources_from_specs(&specs, &registry).await.unwrap();

        assert_eq!(sources.len(), 4);
        assert_eq!(total, 4);
        assert_eq!(
            state
                .max_in_flight
                .load(std::sync::atomic::Ordering::SeqCst),
            4,
            "all populate_remote calls should be in flight simultaneously"
        );
    }

    /// Mixed local/remote specs keep input order and attach the remote tag
    /// only to the remote sources (populated by the concurrent pass).
    #[tokio::test]
    async fn remote_extensions_populated_in_input_order_mixed_specs() {
        let tmp = TempDir::new().unwrap();
        let local = tmp.path().join("b.foo");
        fs::write(&local, b"xy").unwrap();
        let repo = crate::hf_url::remote_repo_for_tests(RepoKind::Model, "order/repo");
        let specs = vec![
            remote_spec(&repo, "a.foo", 1),
            local_spec(&local),
            remote_spec(&repo, "c.foo", 3),
        ];
        let (plugin, _state) = ProbePlugin::new("foo");
        let registry = Registry {
            formats: vec![Arc::new(plugin)],
            ..Registry::default()
        };

        let (sources, total) = prepare_sources_from_specs(&specs, &registry).await.unwrap();

        assert_eq!(sources.len(), 3);
        assert_eq!(total, 6);
        assert!(matches!(sources[0].kind, SourceKind::Http(_)));
        assert_eq!(sources[0].extensions.get::<Tag>().unwrap().0, "remote");
        assert!(matches!(sources[1].kind, SourceKind::File(_)));
        assert_eq!(sources[1].extensions.get::<Tag>().unwrap().0, "local");
        assert!(matches!(sources[2].kind, SourceKind::Http(_)));
        assert_eq!(sources[2].extensions.get::<Tag>().unwrap().0, "remote");
        for (i, s) in sources.iter().enumerate() {
            assert_eq!(s.file_idx, i);
        }
    }

    /// `download_specs_to_paths` passes `filename` to the `hf` CLI as a
    /// positional argument. A repo-level `hf://owner/repo` expands to one
    /// spec per Hub tree entry, so a hostile repo controls that filename;
    /// a name starting with `-` must be rejected before the CLI is spawned
    /// (it would be parsed as a flag, not a path).
    #[tokio::test]
    async fn dash_prefixed_filename_is_rejected_before_hf_is_invoked() {
        let repo = crate::hf_url::remote_repo_for_tests(RepoKind::Model, "evil/repo");
        let spec = crate::hf_url::RemoteFileSpec {
            repo,
            filename: Arc::new("--force-download".to_string()),
            revision: Arc::new("main".to_string()),
            size: 1,
            xet_hash: None,
        };
        let err = download_specs_to_paths(&[spec], "test").await.unwrap_err();
        assert!(
            err.to_string().contains("starts with `-`"),
            "expected the dash-filename rejection, got: {err}"
        );
    }

    fn plain_source(kind: SourceKind) -> Source {
        Source {
            file_idx: 0,
            kind,
            byte_size: 0,
            name_override: None,
            xet_terms: None,
            extensions: Extensions::default(),
        }
    }

    /// Every non-Http source must end up with `xet_terms = Some(vec![])`
    /// ("xet vis requested, but this source isn't xet-backed"), never `None`
    /// (which means "didn't fetch"). This path does no network I/O.
    #[tokio::test]
    async fn populate_xet_terms_marks_non_http_sources_as_empty_not_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.bin");
        fs::write(&path, b"x").unwrap();

        let mut sources = vec![
            plain_source(SourceKind::File(path)),
            plain_source(SourceKind::Buffered(vec![1, 2, 3])),
            plain_source(SourceKind::Diff {
                original: dir.path().join("a.bin"),
                modified: dir.path().join("b.bin"),
            }),
        ];

        populate_xet_terms(&mut sources).await.unwrap();

        for s in &sources {
            assert!(matches!(s.xet_terms, Some(ref v) if v.is_empty()));
        }
    }

    /// With no `SourceKind::Http` entries the materialisation step must be a
    /// complete no-op: Ok, and nothing about the sources changes.
    #[tokio::test]
    async fn materialize_http_sources_is_a_noop_without_http_sources() {
        let mut sources = vec![plain_source(SourceKind::Buffered(vec![9]))];
        let before_name = sources[0].name();

        materialize_http_sources(&mut sources).await.unwrap();

        assert!(matches!(sources[0].kind, SourceKind::Buffered(_)));
        assert!(sources[0].name_override.is_none());
        assert_eq!(sources[0].name(), before_name);
    }

    /// A download failure must propagate as an Err and leave the source list
    /// untouched (kind still Http, no name_override written) — the Http kind
    /// stays intact so a caller can retry or report the real URL. Uses the
    /// dash-filename spec so the failure happens before any CLI spawn.
    #[tokio::test]
    async fn materialize_http_sources_propagates_download_error_and_leaves_sources_unchanged() {
        let repo = crate::hf_url::remote_repo_for_tests(RepoKind::Model, "evil/repo");
        let spec = crate::hf_url::RemoteFileSpec {
            repo,
            filename: Arc::new("--evil".to_string()),
            revision: Arc::new("main".to_string()),
            size: 1,
            xet_hash: None,
        };
        let mut sources = vec![plain_source(SourceKind::Http(spec))];

        let err = materialize_http_sources(&mut sources).await.unwrap_err();
        assert!(
            err.to_string().contains("starts with `-`"),
            "expected the dash-filename rejection, got: {err}"
        );
        assert!(matches!(sources[0].kind, SourceKind::Http(_)));
        assert!(sources[0].name_override.is_none());
        assert!(sources[0].xet_terms.is_none());
    }
}
