//! Option-slot hook impls on top of arbvis's [`Registry`].
//!
//! These wrap the heavy tensor-aware helpers in [`crate::data`] and
//! [`crate::finetune`] into the trait objects arbvis::run dispatches
//! through. Each is a thin glue layer: argument shuffling, error
//! re-contextualisation, no logic of its own. The actual work lives in
//! `crate::data::*`.
//!
//! Wired up by [`crate::register_all`].

use std::path::{Path, PathBuf};

use arbvis::hf_url::RemoteFileSpec;
use arbvis::{
    DiffMetric, DirectoryTensorDiffPrep, FinetuneDetect, LayoutShape, MoeDiffPrep,
    PrepareSourcesExtension, RepoDiffPrep, SingleImageArchHook, Source,
};
use async_trait::async_trait;

use crate::data::{
    build_multi_safetensors_diff_sources, load_meta_for_sources, prepare_diff_sources_from_http,
    prepare_moe_diff_sources,
};
use crate::format::SourceFormat;

/// `--moe-diff <model>` source preparer. Delegates to
/// [`prepare_moe_diff_sources`], which builds the N×N expert-vs-expert
/// diff buffers + `MoeCell` extension tags that
/// [`crate::MoeDiffLayoutPlugin`] reads back.
pub struct TensorMoeDiffPrep;

#[async_trait(?Send)]
impl MoeDiffPrep for TensorMoeDiffPrep {
    async fn prepare(
        &self,
        input: &str,
        metric: DiffMetric,
        stream: bool,
    ) -> anyhow::Result<(Vec<Source>, u64)> {
        prepare_moe_diff_sources(input, metric, stream).await
    }
}

/// Repo-level `--diff hf://... hf://...` preparer. Routes the file
/// listing through [`prepare_diff_sources_from_http`], which lazily
/// diffs safetensors shards over HTTP range requests and eagerly byte-
/// diffs the small non-safetensors siblings (config.json, tokenizer.*).
pub struct TensorRepoDiffPrep;

#[async_trait(?Send)]
impl RepoDiffPrep for TensorRepoDiffPrep {
    async fn prepare(
        &self,
        orig_specs: &[(String, RemoteFileSpec)],
        mod_specs: &[(String, RemoteFileSpec)],
        is_finetune: bool,
        metric: DiffMetric,
        stream: bool,
    ) -> anyhow::Result<(Vec<Source>, u64)> {
        prepare_diff_sources_from_http(orig_specs, mod_specs, is_finetune, metric, stream).await
    }
}

/// Directory `--diff <dir> <dir>` tensor-file preparer. arbvis's
/// directory walk hands us only the entries this hook's
/// [`Self::is_tensor_file`] approved; we run them through
/// [`build_multi_safetensors_diff_sources`] (the multi-shard
/// safetensors / GGUF diff path) and return the resulting per-tensor
/// sources.
pub struct TensorDirectoryDiffPrep;

#[async_trait(?Send)]
impl DirectoryTensorDiffPrep for TensorDirectoryDiffPrep {
    fn is_tensor_file(&self, p: &Path) -> bool {
        SourceFormat::from_path(p).is_some()
    }
    async fn prepare(
        &self,
        orig_files: &[PathBuf],
        mod_files: &[PathBuf],
        is_finetune: bool,
        metric: DiffMetric,
    ) -> anyhow::Result<(Vec<Source>, u64)> {
        build_multi_safetensors_diff_sources(orig_files, mod_files, is_finetune, metric).await
    }
}

/// HF model-card finetune auto-detection. Wraps the pure-async lookup
/// in [`crate::finetune::detect_relation`].
pub struct HfModelCardFinetuneDetect;

#[async_trait(?Send)]
impl FinetuneDetect for HfModelCardFinetuneDetect {
    async fn detect(&self, orig_url: &str, mod_url: &str) -> Option<bool> {
        crate::finetune::detect_relation(orig_url, mod_url).await
    }
}

/// Cross-source sidecar enrichment hook. Runs after `prepare_sources` /
/// `prepare_sources_from_specs` has built every `Source` and per-source
/// `FormatPlugin::populate_*` has stuffed each source's own `ModelInfo`
/// into its `extensions`. We then opportunistically fetch `config.json`
/// and `model.safetensors.index.json` alongside every source (deduped by
/// HF repo + revision or by parent directory) and insert a `SourceMeta`
/// into each source's extensions. [`crate::ArchLayoutPlugin`] reads it
/// back to validate transformer hyperparameters and reserve canonical
/// slots for tensors that live in shards we didn't load.
///
/// Errors from the sidecar fetches are swallowed inside
/// [`crate::data::try_load_source_meta`] — sidecar info is advisory and a
/// missing sidecar must not break rendering — so this hook can't fail.
pub struct SourceMetaSidecarHook;

#[async_trait(?Send)]
impl PrepareSourcesExtension for SourceMetaSidecarHook {
    async fn enrich(&self, sources: &mut [Source]) -> anyhow::Result<()> {
        let metas = load_meta_for_sources(sources).await;
        for (s, m) in sources.iter_mut().zip(metas) {
            s.extensions.insert(m);
        }
        Ok(())
    }
}

/// Single-image arch-layout renderer hook. arbvis's
/// `single::run_single` invokes this when the chosen layout's id is
/// `"arch"`. We delegate to the model-side single-image renderer that
/// lifts the per-tensor color buffers from the `ArchLayout` and paints
/// them via the dtype-aware element colorizer.
pub struct ArchSingleImageHook;

impl SingleImageArchHook for ArchSingleImageHook {
    fn render(
        &self,
        files: &[PathBuf],
        output: Option<PathBuf>,
        sources: &[Source],
        layout: &dyn LayoutShape,
    ) -> anyhow::Result<()> {
        crate::single_arch::run_single_arch(files, output, sources, layout)
    }
}
