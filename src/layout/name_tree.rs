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

/// `(model|transformer|backbone)?\.?(layers|h|blocks|encoder.layer|decoder.layer)\.(\d+)\.(.*)`
/// matches the conventional naming for repeated transformer blocks across
/// llama-family, gpt-family, bert-family, and t5-family checkpoints.
fn block_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^(?:model\.|transformer\.|backbone\.|module\.)?(?:layers|h|blocks|encoder\.layer|decoder\.layer)\.(\d+)\.(.+)$",
        )
        .expect("static regex compiles")
    })
}

/// Classify every tensor name. Returns an `ArchProfile` describing the
/// detected structure.
pub fn classify(names: &[&str]) -> ArchProfile {
    let prefix = detect_common_prefix(names);
    let re = block_regex();

    let mut block_matches = 0usize;
    let mut max_idx: u32 = 0;
    let mut slots: Vec<LayerSlot> = Vec::with_capacity(names.len());

    for &n in names {
        if let Some(caps) = re.captures(n) {
            let idx: u32 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            let sub = caps.get(2).unwrap().as_str().to_string();
            block_matches += 1;
            if idx > max_idx {
                max_idx = idx;
            }
            slots.push(LayerSlot::Block { idx, sub_path: sub });
        } else {
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
        assert_eq!(caps.get(1).unwrap().as_str(), "5");
        assert_eq!(caps.get(2).unwrap().as_str(), "attention.self.query.weight");
    }
}
