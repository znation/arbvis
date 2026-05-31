//! Tensor-name parsing and architectural grouping.
//!
//! Detects transformer-style layouts (`{prefix}.layers.{N}.{sub_path}`) and
//! falls back to a generic name-tree grouping for everything else.

use regex::Regex;
use std::sync::OnceLock;

/// Where a tensor lives in the model's structural hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayerSlot {
    /// Top-level singleton (embed_tokens, lm_head, final norm, …). Identified
    /// by its full tensor name suffix, after stripping any common architecture
    /// prefix shared across the whole checkpoint.
    TopLevel { name: String },
    /// One of the repeated transformer blocks. `idx` is the layer index;
    /// `sub_path` is the dot-separated path inside the block
    /// (e.g. `self_attn.q_proj.weight`).
    Block { idx: u32, sub_path: String },
    /// Generic group (non-transformer architectures). `group` is a dot-path
    /// prefix; `leaf` is the remaining tail.
    Generic { group: String, leaf: String },
}

/// Result of classifying every tensor in a checkpoint.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ArchProfile {
    /// `Some(re)` when the checkpoint was identified as transformer-style.
    /// The regex matches every transformer-block tensor; non-matching tensors
    /// were classified as `TopLevel`.
    pub block_regex: Option<&'static Regex>,
    /// Common architecture prefix stripped from each tensor name (e.g. `model.`).
    /// Empty when no common prefix is found.
    pub prefix: String,
    /// Number of distinct layer indices observed.
    pub num_layers: u32,
    /// Per-tensor classification, parallel to the input name list.
    pub slots: Vec<LayerSlot>,
}

/// Try to identify a common architecture prefix. Looks at every tensor name
/// in `names`; if a prefix like `model.` or `transformer.` is shared by
/// substantially all (>= 80%) of them, returns that prefix (with trailing dot).
/// Otherwise returns empty string.
fn detect_common_prefix(names: &[&str]) -> String {
    let candidates = ["model.", "transformer.", "backbone.", "module."];
    let total = names.len();
    if total == 0 {
        return String::new();
    }
    for cand in &candidates {
        let n = names.iter().filter(|nm| nm.starts_with(cand)).count();
        if n * 5 >= total * 4 {
            return cand.to_string();
        }
    }
    String::new()
}

/// Public accessor for the transformer-block regex. Lets other modules
/// (e.g. the architectural layout) project tensor names into
/// (layer_idx, sub_path) using the same canonical pattern.
pub fn block_regex_for_arch() -> &'static Regex {
    block_regex()
}

/// Matches transformer-block tensor names like `prefix.layers.N.sub_path` for
/// arbitrary dotted prefixes — so the standard llama/gpt/bert/t5 patterns
/// (`model.layers.N.*`, `transformer.h.N.*`, `encoder.layer.N.*`) all match,
/// nested wrappers like `model.language_model.layers.N.*` (Qwen3.5-style VLMs)
/// and bare `mtp.layers.0.*` sidecar streams also match, and the llama.cpp /
/// GGUF convention `blk.N.*` (e.g. `blk.0.attn_q.weight`) too. Pick the
/// dominant prefix in [`classify`] to avoid conflating sidecars with the
/// main stack.
///
/// Captures:
///   1. block-prefix with trailing dot (possibly empty, e.g. `model.language_model.`),
///   2. layer index,
///   3. in-layer sub-path.
fn block_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^((?:[A-Za-z_][A-Za-z_0-9]*\.)*)(?:layers|h|blocks|encoder\.layer|decoder\.layer|blk)\.(\d+)\.(.+)$",
        )
        .expect("static regex compiles")
    })
}

/// Classify every tensor name. Returns an `ArchProfile` describing the
/// detected structure.
pub fn classify(names: &[&str]) -> ArchProfile {
    let prefix = detect_common_prefix(names);
    let re = block_regex();

    // First pass: pull the (block_prefix, idx, sub_path) out of every name
    // that matches the regex. Names that don't match get a `None`.
    let captures: Vec<Option<(String, u32, String)>> = names
        .iter()
        .map(|n| {
            re.captures(n).map(|caps| {
                let bp = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let idx: u32 = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
                let sub = caps.get(3).unwrap().as_str().to_string();
                (bp, idx, sub)
            })
        })
        .collect();

    // Pick the dominant block-prefix: the one with the most distinct layer
    // indices (tie-break: most total tensor matches). This is how we keep
    // sidecar layer streams like Qwen3.5's `mtp.layers.0.*` from poisoning
    // the main transformer stack (`model.language_model.layers.{0..23}.*`):
    // only tensors under the dominant prefix become `Block`s; the rest fall
    // through to `TopLevel`.
    let mut by_prefix: std::collections::HashMap<String, (std::collections::HashSet<u32>, usize)> =
        std::collections::HashMap::new();
    for c in captures.iter().flatten() {
        let entry = by_prefix.entry(c.0.clone()).or_default();
        entry.0.insert(c.1);
        entry.1 += 1;
    }
    let dominant_prefix: Option<String> = by_prefix
        .iter()
        .max_by_key(|(_, (idxs, total))| (idxs.len(), *total))
        .map(|(bp, _)| bp.clone());

    let mut block_matches = 0usize;
    let mut max_idx: u32 = 0;
    let mut slots: Vec<LayerSlot> = Vec::with_capacity(names.len());

    for (&n, c) in names.iter().zip(captures) {
        match c {
            Some((bp, idx, sub)) if Some(&bp) == dominant_prefix.as_ref() => {
                block_matches += 1;
                if idx > max_idx {
                    max_idx = idx;
                }
                slots.push(LayerSlot::Block { idx, sub_path: sub });
            }
            _ => {
                let stripped = if !prefix.is_empty() && n.starts_with(&prefix) {
                    &n[prefix.len()..]
                } else {
                    n
                };
                slots.push(LayerSlot::TopLevel {
                    name: stripped.to_string(),
                });
            }
        }
    }

    // Heuristic: declare "transformer-style" if at least 20% of tensors look
    // like transformer-block parameters AND we saw at least 2 distinct layer
    // indices. Otherwise fall back to generic grouping by name-tree.
    let is_transformer = block_matches * 5 >= names.len() && max_idx >= 1;

    if is_transformer {
        ArchProfile {
            block_regex: Some(re),
            prefix,
            num_layers: max_idx + 1,
            slots,
        }
    } else {
        // Generic name-tree grouping: split each name into (group, leaf) where
        // group is everything up to the last dot. Singleton groups (one tensor)
        // become TopLevel; multi-tensor groups become Generic.
        let mut group_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        let split: Vec<(String, String)> = names
            .iter()
            .map(|n| {
                let stripped = if !prefix.is_empty() && n.starts_with(&prefix) {
                    &n[prefix.len()..]
                } else {
                    n
                };
                match stripped.rfind('.') {
                    Some(p) => (stripped[..p].to_string(), stripped[p + 1..].to_string()),
                    None => (String::new(), stripped.to_string()),
                }
            })
            .collect();
        for (g, _) in &split {
            *group_counts.entry(g.clone()).or_insert(0) += 1;
        }
        let slots = split
            .into_iter()
            .map(|(g, l)| {
                if g.is_empty() || group_counts.get(&g).copied().unwrap_or(0) <= 1 {
                    LayerSlot::TopLevel {
                        name: if g.is_empty() { l } else { format!("{g}.{l}") },
                    }
                } else {
                    LayerSlot::Generic { group: g, leaf: l }
                }
            })
            .collect();
        ArchProfile {
            block_regex: None,
            prefix,
            num_layers: 0,
            slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_qwen_style_llm() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];
        let p = classify(&names);
        assert!(
            p.block_regex.is_some(),
            "expected transformer classification"
        );
        assert_eq!(p.num_layers, 2);
        assert_eq!(p.prefix, "model.");

        let block_count = p
            .slots
            .iter()
            .filter(|s| matches!(s, LayerSlot::Block { .. }))
            .count();
        let top_count = p
            .slots
            .iter()
            .filter(|s| matches!(s, LayerSlot::TopLevel { .. }))
            .count();
        assert_eq!(block_count, 18);
        // embed_tokens, norm, lm_head — lm_head has no `model.` prefix so prefix-stripping is a no-op for it.
        assert_eq!(top_count, 3);
    }

    #[test]
    fn classifies_gpt2_h_style() {
        let names = vec![
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.1.ln_1.weight",
            "transformer.h.1.attn.c_attn.weight",
            "transformer.h.1.mlp.c_fc.weight",
            "transformer.ln_f.weight",
        ];
        let p = classify(&names);
        assert!(p.block_regex.is_some());
        assert_eq!(p.num_layers, 2);
        assert_eq!(p.prefix, "transformer.");
    }

    #[test]
    fn falls_back_to_generic_for_non_transformer() {
        let names = vec![
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_in.bias",
            "first_stage_model.decoder.conv_out.weight",
            "first_stage_model.decoder.conv_out.bias",
        ];
        let p = classify(&names);
        assert!(
            p.block_regex.is_none(),
            "should not classify as transformer"
        );
    }

    #[test]
    fn asymmetric_prefix_uses_majority() {
        // Mixed prefixes; "model." wins because it's the supermajority.
        let mut names: Vec<&'static str> = Vec::new();
        for i in 0..20 {
            names.push(Box::leak(
                format!("model.layers.{i}.weight").into_boxed_str(),
            ));
        }
        names.push("lm_head.weight");
        names.push("model.norm.weight");
        let p = classify(&names);
        assert_eq!(p.prefix, "model.");
        assert!(p.block_regex.is_some());
    }

    #[test]
    fn single_tensor_no_group() {
        let names = vec!["foo.weight"];
        let p = classify(&names);
        assert!(p.block_regex.is_none());
        assert_eq!(p.slots.len(), 1);
        // No common prefix to strip.
        match &p.slots[0] {
            LayerSlot::TopLevel { name } => assert_eq!(name, "foo.weight"),
            other => panic!("unexpected slot {other:?}"),
        }
    }

    #[test]
    fn block_regex_matches_bert_encoder_layer() {
        let re = block_regex();
        let caps = re
            .captures("encoder.layer.5.attention.self.query.weight")
            .unwrap();
        // Group 1 = block-prefix (empty here, since `encoder.layer` is itself
        // the block keyword); group 2 = layer idx; group 3 = sub-path.
        assert_eq!(caps.get(1).unwrap().as_str(), "");
        assert_eq!(caps.get(2).unwrap().as_str(), "5");
        assert_eq!(caps.get(3).unwrap().as_str(), "attention.self.query.weight");
    }

    #[test]
    fn block_regex_matches_nested_prefix() {
        // Qwen3.5-style multimodal wrapper: `model.language_model.layers.N.*`.
        // The block regex must walk through arbitrary dotted prefixes.
        let re = block_regex();
        let caps = re
            .captures("model.language_model.layers.12.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(caps.get(1).unwrap().as_str(), "model.language_model.");
        assert_eq!(caps.get(2).unwrap().as_str(), "12");
        assert_eq!(caps.get(3).unwrap().as_str(), "self_attn.q_proj.weight");
    }

    #[test]
    fn classifies_nested_prefix_as_blocks() {
        // 24-layer language model under a `model.language_model.` wrapper.
        // Without nested-prefix support, none of these would match the regex
        // and every tensor would fall to TopLevel — losing the architectural
        // layout entirely.
        let mut owned: Vec<String> = vec![
            "model.language_model.embed_tokens.weight".to_string(),
            "model.language_model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for i in 0..24 {
            owned.push(format!(
                "model.language_model.layers.{i}.self_attn.q_proj.weight"
            ));
            owned.push(format!(
                "model.language_model.layers.{i}.self_attn.k_proj.weight"
            ));
            owned.push(format!(
                "model.language_model.layers.{i}.mlp.gate_proj.weight"
            ));
        }
        let names: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let p = classify(&names);
        assert!(
            p.block_regex.is_some(),
            "nested-prefix model should classify as transformer"
        );
        assert_eq!(p.num_layers, 24);
        let block_count = p
            .slots
            .iter()
            .filter(|s| matches!(s, LayerSlot::Block { .. }))
            .count();
        assert_eq!(block_count, 24 * 3);
    }

    #[test]
    fn dominant_prefix_wins_over_sidecar_stream() {
        // Qwen3.5 ships an MTP sidecar with one `mtp.layers.0.*` block
        // alongside the main 24-layer `model.language_model.layers.{0..23}.*`
        // stack. If we naively treated every regex match as a Block, the
        // mtp tensors would collide with layer 0 of the main stack. Instead,
        // the dominant prefix (most layer indices) wins, and the sidecar
        // tensors fall through to TopLevel.
        let mut owned: Vec<String> = Vec::new();
        for i in 0..24 {
            owned.push(format!(
                "model.language_model.layers.{i}.self_attn.q_proj.weight"
            ));
        }
        // Sidecar — single layer index, different prefix.
        owned.push("mtp.layers.0.self_attn.q_proj.weight".to_string());
        owned.push("mtp.layers.0.mlp.gate_proj.weight".to_string());
        let names: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let p = classify(&names);
        assert_eq!(p.num_layers, 24, "main stack should have 24 layers");
        // The mtp tensors must NOT have been folded into the main stack: they
        // should land as TopLevel rather than Block.
        let mtp_slots: Vec<&LayerSlot> = p
            .slots
            .iter()
            .zip(names.iter())
            .filter(|(_, n)| n.starts_with("mtp."))
            .map(|(s, _)| s)
            .collect();
        assert_eq!(mtp_slots.len(), 2);
        for s in mtp_slots {
            assert!(
                matches!(s, LayerSlot::TopLevel { .. }),
                "sidecar tensor should be TopLevel, got {s:?}",
            );
        }
    }

    #[test]
    fn block_regex_matches_gguf_blk() {
        // GGUF uses bare `blk.N.*` with no architecture prefix — capture group
        // 1 (block-prefix) is empty, group 2 is the layer idx, group 3 is the
        // in-layer sub-path.
        let re = block_regex();
        let caps = re.captures("blk.0.attn_q.weight").unwrap();
        assert_eq!(caps.get(1).unwrap().as_str(), "");
        assert_eq!(caps.get(2).unwrap().as_str(), "0");
        assert_eq!(caps.get(3).unwrap().as_str(), "attn_q.weight");
        let caps = re.captures("blk.31.ffn_down.weight").unwrap();
        assert_eq!(caps.get(1).unwrap().as_str(), "");
        assert_eq!(caps.get(2).unwrap().as_str(), "31");
        assert_eq!(caps.get(3).unwrap().as_str(), "ffn_down.weight");
    }

    #[test]
    fn classifies_qwen3_gguf_style_llm() {
        let names = vec![
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_k.weight",
            "blk.1.attn_v.weight",
            "blk.1.attn_output.weight",
            "blk.1.ffn_gate.weight",
            "blk.1.ffn_up.weight",
            "blk.1.ffn_down.weight",
            "blk.1.attn_norm.weight",
            "blk.1.ffn_norm.weight",
            "output_norm.weight",
            "output.weight",
        ];
        let p = classify(&names);
        assert!(
            p.block_regex.is_some(),
            "expected transformer classification"
        );
        assert_eq!(p.num_layers, 2);
        let block_count = p
            .slots
            .iter()
            .filter(|s| matches!(s, LayerSlot::Block { .. }))
            .count();
        assert_eq!(block_count, 18);
    }
}
