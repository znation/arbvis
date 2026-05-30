//! Cross-format tensor-name canonicalisation.
//!
//! Safetensors uses HuggingFace-style names like
//! `model.layers.5.self_attn.q_proj.weight`. GGUF uses llama.cpp-style names
//! like `blk.5.attn_q.weight`. The diff matcher needs to pair tensors
//! across formats; we do that by normalizing both sides to the same
//! canonical form (HF-style) before running the strip heuristic.
//!
//! This is a minimal v1 table covering the llama / qwen / mistral families.
//! It mirrors the relevant subset of llama.cpp's `gguf-py/gguf/tensor_mapping.py`.
//! Future formats can extend the table or short-circuit in `to_canonical`.

use super::SourceFormat;

/// Normalize `name` to its canonical HF-style form. For safetensors and
/// PyTorch pickle this is a no-op (both use the HF state_dict naming
/// convention directly). For GGUF this maps `blk.N.attn_q.weight` →
/// `model.layers.N.self_attn.q_proj.weight` and similar.
pub fn to_canonical(format: SourceFormat, name: &str) -> String {
    match format {
        SourceFormat::Safetensors | SourceFormat::Pickle => name.to_string(),
        SourceFormat::Gguf => gguf_to_hf(name),
    }
}

/// Map a GGUF tensor name to its HF safetensors equivalent. Falls back to
/// the input name if no rule matches — the strip heuristic will still get
/// a chance to match identical-named tensors (uncommon, but harmless).
fn gguf_to_hf(name: &str) -> String {
    // Top-level singletons.
    match name {
        "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
        "output_norm.weight" => return "model.norm.weight".to_string(),
        "output.weight" => return "lm_head.weight".to_string(),
        "rope_freqs.weight" => return "model.rotary_emb.inv_freq".to_string(),
        _ => {}
    }

    // Block-scoped (`blk.{N}.{sub}`).
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some(dot) = rest.find('.') else {
        return name.to_string();
    };
    let layer = &rest[..dot];
    let sub = &rest[dot + 1..];
    let translated_sub = match sub {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        // Qwen3 MoE experts. The HF side stores per-expert as
        // `mlp.experts.{i}.{gate|up|down}_proj.weight`, but the GGUF side
        // fuses all experts into one tensor named `ffn_{gate|up|down}_exps.weight`.
        // For canonicalisation we collapse both to the GGUF fused form so
        // the matcher pairs them as one tensor (the diff itself runs over
        // bytes, not per-expert).
        "ffn_gate_exps.weight" => "mlp.experts.gate_proj.weight",
        "ffn_up_exps.weight" => "mlp.experts.up_proj.weight",
        "ffn_down_exps.weight" => "mlp.experts.down_proj.weight",
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        // Pass-through anything we don't recognise.
        other => return format!("model.layers.{layer}.{other}"),
    };
    format!("model.layers.{layer}.{translated_sub}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_to_hf_block_attn() {
        assert_eq!(
            gguf_to_hf("blk.5.attn_q.weight"),
            "model.layers.5.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_to_hf("blk.0.attn_output.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn gguf_to_hf_block_ffn() {
        assert_eq!(
            gguf_to_hf("blk.10.ffn_gate.weight"),
            "model.layers.10.mlp.gate_proj.weight"
        );
        assert_eq!(
            gguf_to_hf("blk.10.ffn_down.weight"),
            "model.layers.10.mlp.down_proj.weight"
        );
    }

    #[test]
    fn gguf_to_hf_top_level() {
        assert_eq!(gguf_to_hf("token_embd.weight"), "model.embed_tokens.weight");
        assert_eq!(gguf_to_hf("output.weight"), "lm_head.weight");
        assert_eq!(gguf_to_hf("output_norm.weight"), "model.norm.weight");
    }

    #[test]
    fn gguf_to_hf_unknown_passes_through() {
        // Unknown blk sub-path: keep the layer prefix but pass the leaf through.
        assert_eq!(
            gguf_to_hf("blk.3.something.weight"),
            "model.layers.3.something.weight"
        );
        // Entirely unknown name: identity.
        assert_eq!(gguf_to_hf("custom.tensor"), "custom.tensor");
    }

    #[test]
    fn safetensors_to_canonical_is_identity() {
        assert_eq!(
            to_canonical(
                SourceFormat::Safetensors,
                "model.layers.0.self_attn.q_proj.weight"
            ),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn cross_format_matches_after_canonical() {
        let st = to_canonical(
            SourceFormat::Safetensors,
            "model.layers.5.self_attn.q_proj.weight",
        );
        let gg = to_canonical(SourceFormat::Gguf, "blk.5.attn_q.weight");
        assert_eq!(st, gg);
    }
}
