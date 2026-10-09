//! arbvis's two built-in [`SourceProvider`]s, installed by
//! [`Registry::with_defaults`], plus provider selection. A specialization
//! registers its own higher-priority providers; these are the always-present
//! byte fallbacks.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};

use crate::data::{self, InputSpec, Source};
use crate::hf_url;
use crate::registry::{Registry, RenderHints, SourceCtx, SourceProvider};

/// Concurrency cap when resolving (downloading) `hf://` inputs at startup.
/// Mirrors `data::SETUP_FETCH_CONCURRENCY` so user-visible parallelism stays
/// consistent across the input-resolution and materialisation stages.
const RESOLVE_CONCURRENCY: usize = 16;

/// Byte/JSON diff over a `--diff` pair (priority 100). Resolves both sides
/// (local path or single-file `hf://`) and dispatches through the file-pair
/// builder cascade or [`crate::data_diff::byte_directory_diff`].
pub(crate) struct ByteDiffProvider;

#[async_trait(?Send)]
impl SourceProvider for ByteDiffProvider {
    fn id(&self) -> &'static str {
        "byte-diff"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
        ctx.diff.is_some()
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        let diff = ctx
            .diff
            .as_ref()
            .expect("ByteDiffProvider applies only when --diff is set");
        // Resolve both sides concurrently; the (original, modified) order is
        // part of the diff contract, so `try_join!` (not unordered) is used.
        let (orig, mod_) = tokio::try_join!(
            resolve_input(PathBuf::from(diff.original)),
            resolve_input(PathBuf::from(diff.modified)),
        )?;
        let (sources, total) =
            crate::data_diff::prepare_diff_sources(&orig, &mod_, false, ctx.registry).await?;
        let hints = RenderHints {
            diff_mode: true,
            title_suffix: std::borrow::Cow::Borrowed("diff"),
            show_xet_xorbs: false,
            inputs: vec![diff.original.to_string(), diff.modified.to_string()],
        };
        Ok((sources, total, hints))
    }
}

/// Normal byte render of the positional inputs (or stdin). The priority floor
/// (`i32::MIN`) — always applicable, so provider selection always terminates.
pub(crate) struct NormalBytesProvider;

#[async_trait(?Send)]
impl SourceProvider for NormalBytesProvider {
    fn id(&self) -> &'static str {
        "normal-bytes"
    }
    fn priority(&self) -> i32 {
        i32::MIN
    }
    fn applicable(&self, _ctx: &SourceCtx<'_>) -> bool {
        true
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        let (sources, total) =
            resolve_input_sources(ctx.inputs, ctx.show_xet_xorbs, ctx.stream, ctx.registry).await?;
        let inputs: Vec<String> = ctx
            .inputs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let hints = RenderHints {
            diff_mode: false,
            title_suffix: std::borrow::Cow::Borrowed(""),
            show_xet_xorbs: ctx.show_xet_xorbs,
            inputs,
        };
        Ok((sources, total, hints))
    }
}

/// Pick the highest-priority [`SourceProvider`] whose `applicable` returns true.
/// Sorted descending by priority; the `i32::MIN` floor (`NormalBytesProvider`)
/// guarantees a result for any registry built via [`Registry::with_defaults`].
pub(crate) fn select_provider<'a>(
    providers: &'a [Arc<dyn SourceProvider>],
    ctx: &SourceCtx<'_>,
) -> Option<&'a Arc<dyn SourceProvider>> {
    let mut sorted: Vec<&Arc<dyn SourceProvider>> = providers.iter().collect();
    sorted.sort_by_key(|p| std::cmp::Reverse(p.priority()));
    sorted.into_iter().find(|p| p.applicable(ctx))
}

/// Build `Source`s for the normal (non-diff) flow.
///
/// In disk-backed mode (the default), every `hf://` input is downloaded to
/// the local HF cache (via the `hf` CLI) by [`data::materialize_http_sources`]. In
/// `--stream` mode the sources stay remote and per-tile reads hit HTTP
/// directly; that's only useful when inputs don't fit on local disk and is
/// substantially slower otherwise. `--show-xet-xorbs` captures xet term
/// metadata before materialization (the post-download file lacks the remote
/// spec needed to query it).
pub(crate) async fn resolve_input_sources(
    files: &[PathBuf],
    show_xet_xorbs: bool,
    stream: bool,
    registry: &Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Streaming mode keeps hf:// inputs remote; show-xet-xorbs is the other
    // case that has to start from specs (it needs the remote spec to look up
    // xet metadata). Otherwise we can fast-path through `prepare_sources`
    // which downloads via hf_url::resolve and mmaps the local file.
    if !stream && !show_xet_xorbs {
        let resolved: Vec<PathBuf> =
            futures::stream::iter(files.iter().cloned().map(resolve_input))
                .buffered(RESOLVE_CONCURRENCY)
                .try_collect()
                .await?;
        return data::prepare_sources(&resolved, registry);
    }

    // Resolve `hf://` paths into specs concurrently. Repo-level URLs expand
    // to multiple specs; non-hf:// paths stay as `InputSpec::Local`. The
    // result preserves input order so labels and byte offsets are
    // deterministic across runs.
    let specs: Vec<InputSpec> = futures::stream::iter(files.iter().cloned().map(|p| async move {
        let s = p.to_string_lossy();
        if hf_url::is_hf_url(&s) {
            if hf_url::is_repo_level(&s)? {
                let listed = hf_url::list_repo_as_http_specs(&s)
                    .await
                    .with_context(|| format!("listing files in {s}"))?;
                anyhow::Ok(
                    listed
                        .into_iter()
                        .map(|(_, spec)| InputSpec::Remote(spec))
                        .collect::<Vec<_>>(),
                )
            } else {
                anyhow::Ok(vec![InputSpec::Remote(hf_url::resolve_to_http(&p).await?)])
            }
        } else {
            anyhow::Ok(vec![InputSpec::Local(p)])
        }
    }))
    .buffered(RESOLVE_CONCURRENCY)
    .try_collect::<Vec<_>>()
    .await?
    .into_iter()
    .flatten()
    .collect();
    let (mut sources, total) = data::prepare_sources_from_specs(&specs, registry).await?;
    if show_xet_xorbs {
        data::populate_xet_terms(&mut sources).await?;
    }
    if !stream {
        // Per-range HTTPS GETs are too expensive for the tile workload
        // — one whole-file download amortises the connection setup over the
        // entire file (which we read every byte of anyway during render).
        // See `materialize_http_sources` for the full story.
        data::materialize_http_sources(&mut sources).await?;
    }
    Ok((sources, total))
}

/// Resolve an input path: download from HF if it starts with `hf://`.
async fn resolve_input(path: PathBuf) -> anyhow::Result<PathBuf> {
    let display = path.display().to_string();
    hf_url::resolve(&path)
        .await
        .with_context(|| format!("resolving {display}"))
}

#[cfg(test)]
mod provider_selection_tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::select_provider;
    use crate::registry::{DestKind, DiffPair, Registry, RenderHints, SourceCtx, SourceProvider};

    /// A high-priority diff-only provider, standing in for a downstream
    /// specialization's provider (e.g. modelweightvis's `RepoDiffProvider`).
    struct MockDiffProvider;

    #[async_trait(?Send)]
    impl SourceProvider for MockDiffProvider {
        fn id(&self) -> &'static str {
            "mock-high"
        }
        fn priority(&self) -> i32 {
            500
        }
        fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
            ctx.diff.is_some()
        }
        async fn prepare(
            &self,
            _ctx: &SourceCtx<'_>,
        ) -> anyhow::Result<(Vec<crate::data::Source>, u64, RenderHints)> {
            Ok((Vec::new(), 0, RenderHints::default()))
        }
    }

    fn ctx<'a>(
        reg: &'a Registry,
        inputs: &'a [PathBuf],
        diff: Option<DiffPair<'a>>,
    ) -> SourceCtx<'a> {
        SourceCtx {
            inputs,
            diff,
            dest_kind: DestKind::Bundle,
            three_d: false,
            stream: false,
            show_xet_xorbs: false,
            registry: reg,
        }
    }

    // The byte built-ins: a normal-input invocation falls to the `i32::MIN`
    // floor; a `--diff` invocation is caught by the byte-diff provider.
    #[test]
    fn byte_builtins_ladder() {
        let reg = Registry::with_defaults();
        let inputs = vec![PathBuf::from("a.bin")];
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &inputs, None))
                .unwrap()
                .id(),
            "normal-bytes"
        );
        let none: Vec<PathBuf> = Vec::new();
        let pair = DiffPair {
            original: "a",
            modified: "b",
        };
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &none, Some(pair)))
                .unwrap()
                .id(),
            "byte-diff"
        );
    }

    // A higher-priority provider shadows the byte-diff built-in when it
    // applies, but the floor still wins when it doesn't.
    #[test]
    fn higher_priority_shadows_then_falls_through() {
        let mut reg = Registry::with_defaults();
        reg.providers.push(Arc::new(MockDiffProvider));
        let none: Vec<PathBuf> = Vec::new();
        let pair = DiffPair {
            original: "a",
            modified: "b",
        };
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &none, Some(pair)))
                .unwrap()
                .id(),
            "mock-high"
        );
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &none, None))
                .unwrap()
                .id(),
            "normal-bytes"
        );
    }
}
