//! Tensor-aware `DiffSourceBuilder` impl. Delegates to arbvis's
//! `build_safetensors_diff_sources` helper (still lives in arbvis as a
//! pub-exposed function; step 12e's full source relocation is deferred).

use std::path::Path;

use arbvis::{
    build_safetensors_diff_sources, DiffBuildCtx, DiffSourceBuilder, Source, SourceFormat,
};

/// Tensor-aware diff (safetensors / GGUF). Applies when both paths look
/// like a recognised model-format file.
pub struct TensorDiffBuilder;

#[async_trait::async_trait]
impl DiffSourceBuilder for TensorDiffBuilder {
    fn id(&self) -> &'static str {
        "tensor"
    }
    fn priority(&self) -> i32 {
        300
    }
    async fn try_build(
        &self,
        ctx: &DiffBuildCtx<'_>,
    ) -> anyhow::Result<Option<(Vec<Source>, u64)>> {
        let is_st = |p: &Path| -> bool { SourceFormat::from_path(p).is_some() };
        if !(is_st(ctx.original) && is_st(ctx.modified)) {
            return Ok(None);
        }
        let out =
            build_safetensors_diff_sources(ctx.original, ctx.modified, ctx.is_finetune, ctx.metric)
                .await?;
        Ok(Some(out))
    }
}
