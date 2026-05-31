//! HF model-card "is mod a finetune of orig" auto-detection.
//!
//! Step 11 of the arbvis/modelweightvis split. The logic is model-specific
//! (walks `cardData.base_model` / `base_model_relation` fields produced by
//! the HF transformers training scripts) so it moves to `modelweightvis`
//! wholesale in step 12. For now it lives in arbvis and depends only on
//! [`crate::hf_url`]'s generic `fetch_model_card`.

use arbvis::hf_url::{self, RepoKind};

/// Auto-detect whether `mod_url` is a HuggingFace-declared finetune of
/// `orig_url`, by reading the modified-side model card metadata via the HF
/// Hub API.
///
/// Returns:
/// - `Some(true)` when both args are model URLs **and** the modified side's
///   `cardData.base_model` (string or list) includes the original side's
///   repo id **and** `cardData.base_model_relation` is `finetune` (or
///   unspecified — HF's convention is that relation defaults to finetune
///   when omitted but a base is declared).
/// - `Some(false)` when both args are HF model URLs but the metadata
///   doesn't establish a finetune relation in the orig→mod direction.
/// - `None` when detection isn't applicable: either side isn't an `hf://`
///   model URL, or the API call failed (network error, private repo
///   without token, 404, etc.). Caller falls back to its own default.
pub async fn detect_relation(orig_url: &str, mod_url: &str) -> Option<bool> {
    let orig_hf = hf_url::parse(orig_url).ok()?;
    let mod_hf = hf_url::parse(mod_url).ok()?;
    // The finetune relation is only meaningful between two model repos.
    if !matches!(orig_hf.kind, RepoKind::Model) || !matches!(mod_hf.kind, RepoKind::Model) {
        return None;
    }

    let info = match hf_url::fetch_model_card(&mod_hf.repo_id).await {
        Ok(v) => v,
        Err(e) => {
            log::debug!(
                "finetune auto-detect: HF API lookup for {} failed: {e:#}",
                mod_hf.repo_id
            );
            return None;
        }
    };

    let cd = info.get("cardData")?;
    let base_match = match cd.get("base_model") {
        Some(serde_json::Value::String(s)) => repo_ids_equal(s, &orig_hf.repo_id),
        Some(serde_json::Value::Array(a)) => a.iter().any(|v| {
            v.as_str()
                .map(|s| repo_ids_equal(s, &orig_hf.repo_id))
                .unwrap_or(false)
        }),
        _ => return Some(false),
    };
    if !base_match {
        return Some(false);
    }
    // Per the HF model-card spec the relation defaults to "finetune" when a
    // base is declared but no explicit relation is given. Anything else
    // ("quantized", "merge", "adapter", "other", ...) does *not* satisfy the
    // finetune contract.
    let relation = cd
        .get("base_model_relation")
        .and_then(|v| v.as_str())
        .unwrap_or("finetune");
    Some(relation.eq_ignore_ascii_case("finetune"))
}

/// `owner/name` comparison that tolerates case differences in the owner /
/// name segments (HF treats these as case-insensitive).
fn repo_ids_equal(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}
