//! Mixture-of-Experts tensor name parsing.
//!
//! HuggingFace-style MoE checkpoints store each expert as a distinct tensor:
//!     `model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight`
//!
//! This covers Mixtral, Qwen3-MoE, OLMoE, DeepSeek-V2/V3 (routed experts).
//! DeepSeek-style `mlp.shared_experts.*` are intentionally *not* matched —
//! shared experts run on every token and aren't part of the routed N×N matrix.
//!
//! GGUF fuses all experts of one layer into a single tensor named
//! `blk.{L}.ffn_{gate|up|down}_exps.weight`. Per-expert visualisation of GGUF
//! checkpoints is out of scope for v1 — `is_fused_gguf_expert` lets callers
//! detect and reject this case cleanly.
//!
//! See [`crate::format::name_map`] for the cross-format diff canonicaliser
//! (which deliberately collapses HF per-expert names to the GGUF fused form
//! for the regular `--diff` flow).
//!
//! NB: this module is the parser only. Source construction and matrix layout
//! live in [`crate::data::prepare_moe_diff_sources`] and
//! [`crate::layout::arch::ArchLayout::try_build_moe_diff`].

/// Which of the three FFN weights of a single expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum ExpertWeight {
    GateProj,
    UpProj,
    DownProj,
}

impl ExpertWeight {
    pub fn label(self) -> &'static str {
        match self {
            ExpertWeight::GateProj => "gate_proj",
            ExpertWeight::UpProj => "up_proj",
            ExpertWeight::DownProj => "down_proj",
        }
    }
}

/// A successfully-parsed per-expert tensor reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRef {
    pub layer_idx: u32,
    pub expert_idx: u32,
    pub weight: ExpertWeight,
}

/// Parse an HF-style per-expert tensor name. Returns `None` for any tensor
/// that doesn't match the routed-experts pattern (top-level tensors,
/// non-MoE layers, shared experts, router gates, biases).
pub fn parse_hf_expert(name: &str) -> Option<ExpertRef> {
    // model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer_idx: u32 = layer_str.parse().ok()?;

    // The MoE block prefix is `mlp.experts.` in the layouts we target
    // (Mixtral, Qwen3, OLMoE, DeepSeek routed experts). DeepSeek's
    // `mlp.shared_experts.*` does NOT match — those aren't routed.
    let rest = rest.strip_prefix("mlp.experts.")?;

    let (expert_str, leaf) = rest.split_once('.')?;
    let expert_idx: u32 = expert_str.parse().ok()?;

    let weight = match leaf {
        "gate_proj.weight" => ExpertWeight::GateProj,
        "up_proj.weight" => ExpertWeight::UpProj,
        "down_proj.weight" => ExpertWeight::DownProj,
        _ => return None,
    };

    Some(ExpertRef {
        layer_idx,
        expert_idx,
        weight,
    })
}

/// Whether `name` is a GGUF fused-expert tensor (`ffn_{gate|up|down}_exps.weight`,
/// optionally under `blk.{L}.`). Used by callers to bail with a clear error
/// before attempting per-expert layout on a GGUF checkpoint.
pub fn is_fused_gguf_expert(name: &str) -> bool {
    let leaf = match name.strip_prefix("blk.") {
        Some(rest) => rest.split_once('.').map(|(_, x)| x).unwrap_or(rest),
        None => name,
    };
    matches!(
        leaf,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen3_moe_expert() {
        let r = parse_hf_expert("model.layers.5.mlp.experts.12.gate_proj.weight").unwrap();
        assert_eq!(r.layer_idx, 5);
        assert_eq!(r.expert_idx, 12);
        assert_eq!(r.weight, ExpertWeight::GateProj);

        let r = parse_hf_expert("model.layers.0.mlp.experts.0.up_proj.weight").unwrap();
        assert_eq!(r.layer_idx, 0);
        assert_eq!(r.expert_idx, 0);
        assert_eq!(r.weight, ExpertWeight::UpProj);

        let r = parse_hf_expert("model.layers.31.mlp.experts.63.down_proj.weight").unwrap();
        assert_eq!(r.layer_idx, 31);
        assert_eq!(r.expert_idx, 63);
        assert_eq!(r.weight, ExpertWeight::DownProj);
    }

    #[test]
    fn rejects_shared_experts() {
        // DeepSeek-style shared experts run on every token — not part of
        // the routed N×N matrix.
        assert_eq!(
            parse_hf_expert("model.layers.3.mlp.shared_experts.gate_proj.weight"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.3.mlp.shared_experts.0.gate_proj.weight"),
            None,
        );
    }

    #[test]
    fn rejects_router_and_norms() {
        // Router gate (`mlp.gate.weight` in HF / `ffn_gate_inp.weight` in GGUF).
        assert_eq!(parse_hf_expert("model.layers.0.mlp.gate.weight"), None);
        // Non-MoE dense MLP.
        assert_eq!(parse_hf_expert("model.layers.0.mlp.gate_proj.weight"), None,);
        // Norms / attention / top-level singletons.
        assert_eq!(
            parse_hf_expert("model.layers.0.input_layernorm.weight"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.self_attn.q_proj.weight"),
            None,
        );
        assert_eq!(parse_hf_expert("model.embed_tokens.weight"), None);
        assert_eq!(parse_hf_expert("lm_head.weight"), None);
    }

    #[test]
    fn rejects_non_weight_leaves() {
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.0.gate_proj.bias"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.0.w1.weight"),
            None,
        );
    }

    #[test]
    fn rejects_unparseable_indices() {
        assert_eq!(
            parse_hf_expert("model.layers.x.mlp.experts.0.gate_proj.weight"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.y.gate_proj.weight"),
            None,
        );
    }

    #[test]
    fn detects_gguf_fused_experts() {
        assert!(is_fused_gguf_expert("blk.0.ffn_gate_exps.weight"));
        assert!(is_fused_gguf_expert("blk.31.ffn_up_exps.weight"));
        assert!(is_fused_gguf_expert("blk.15.ffn_down_exps.weight"));
        // Bare-leaf form (canonicaliser strips the `blk.{N}.` prefix
        // before lookup).
        assert!(is_fused_gguf_expert("ffn_gate_exps.weight"));
    }

    #[test]
    fn fused_detector_rejects_unrelated() {
        assert!(!is_fused_gguf_expert("blk.0.ffn_gate.weight"));
        assert!(!is_fused_gguf_expert("blk.0.attn_q.weight"));
        assert!(!is_fused_gguf_expert(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ));
    }
}
