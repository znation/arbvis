//! Opportunistic parsing of HuggingFace `config.json` and
//! `model.safetensors.index.json` sidecar files.
//!
//! Both are advisory — the architectural layout works without them by
//! inferring everything from tensor names + shapes. When they ARE available,
//! they let us:
//!
//! - Validate that the layer count we inferred from `model.layers.N` matches
//!   `num_hidden_layers` (warn on mismatch).
//! - Extend the canonical layer arrangement to cover *all* `num_hidden_layers`
//!   transformer blocks when only a subset of shards was loaded (missing
//!   blocks render as padded slots instead of being skipped, which keeps
//!   layouts stable across partial loads and across the two sides of a diff
//!   that happened to load different shard subsets).
//! - Tag entities in `labels.json` with a model architecture string for the
//!   viewer to display.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file::Value;
use serde::Deserialize;

use crate::format::gguf::{metadata_array_len, metadata_string, metadata_u64};

/// Fields we care about from a HuggingFace `config.json`. Everything is
/// optional because configs vary across architectures — `vocab_size` and
/// `intermediate_size` aren't present on every model card.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelConfig {
    /// `["LlamaForCausalLM", …]` etc. Used for display only.
    #[serde(default)]
    pub architectures: Vec<String>,
    pub num_hidden_layers: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub vocab_size: Option<u64>,
}

impl ModelConfig {
    /// Parse from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        match serde_json::from_slice::<Self>(bytes) {
            Ok(c) => Some(c),
            Err(e) => {
                log::warn!("config.json: parse failed: {e}");
                None
            }
        }
    }

    /// Try to load a `config.json` from `dir`. Returns `None` when the file
    /// is absent or unparsable; logs a warning in the latter case.
    pub fn try_from_dir(dir: &Path) -> Option<Self> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path).ok()?;
        log::debug!("loaded config.json from {}", path.display());
        Self::from_bytes(&bytes)
    }

    /// Short display string: first architecture name + key dims.
    pub fn summary(&self) -> String {
        let arch = self
            .architectures
            .first()
            .map(String::as_str)
            .unwrap_or("?");
        let n = self.num_hidden_layers.unwrap_or(0);
        let h = self.hidden_size.unwrap_or(0);
        format!("{arch} ({n} layers, hidden={h})")
    }

    /// Populate a `ModelConfig` from a parsed GGUF metadata KV table.
    ///
    /// GGUF embeds the equivalent of `config.json` inline; we map the
    /// architecture name to `architectures` and pull the standard
    /// `{arch}.block_count` / `embedding_length` / `attention.head_count` /
    /// `attention.head_count_kv` / `feed_forward_length` keys into the
    /// matching fields. Missing keys remain `None` — the architectural
    /// layout already tolerates that.
    pub fn from_gguf_metadata(metadata: &HashMap<String, Value>) -> Self {
        let arch_name = metadata_string(metadata, "general.architecture").map(String::from);
        let architectures = arch_name.iter().cloned().collect::<Vec<_>>();
        // The per-arch keys are prefixed with the architecture name (e.g.
        // `llama.block_count`, `qwen3.embedding_length`). If we don't know
        // the architecture we still try a few common prefixes.
        let candidates: Vec<&str> = match arch_name.as_deref() {
            Some(a) => vec![a],
            None => vec!["llama", "qwen2", "qwen3", "mistral", "mixtral"],
        };
        let try_key = |suffix: &str| -> Option<u64> {
            candidates
                .iter()
                .find_map(|a| metadata_u64(metadata, &format!("{a}.{suffix}")))
        };
        let num_hidden_layers = try_key("block_count").map(|v| v as u32);
        let hidden_size = try_key("embedding_length").map(|v| v as u32);
        let num_attention_heads = try_key("attention.head_count").map(|v| v as u32);
        let num_key_value_heads = try_key("attention.head_count_kv").map(|v| v as u32);
        let intermediate_size = try_key("feed_forward_length").map(|v| v as u32);
        let vocab_size = metadata_array_len(metadata, "tokenizer.ggml.tokens").map(|n| n as u64);
        ModelConfig {
            architectures,
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            vocab_size,
        }
    }
}

/// Fields we care about from `model.safetensors.index.json`. The map tells us
/// which shard file every tensor lives in — useful for enumerating ALL tensor
/// names when only some shards are loaded.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SafetensorsIndex {
    /// `tensor_name → shard_filename`.
    pub weight_map: std::collections::HashMap<String, String>,
}

impl SafetensorsIndex {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        match serde_json::from_slice::<Self>(bytes) {
            Ok(i) => Some(i),
            Err(e) => {
                log::warn!("model.safetensors.index.json: parse failed: {e}");
                None
            }
        }
    }

    /// Try to load `model.safetensors.index.json` from `dir`. Returns `None`
    /// when the file is absent or unparsable.
    pub fn try_from_dir(dir: &Path) -> Option<Self> {
        let path = dir.join("model.safetensors.index.json");
        let bytes = std::fs::read(&path).ok()?;
        log::debug!("loaded safetensors index from {}", path.display());
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_style_config() {
        let json = br#"{
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256
        }"#;
        let c = ModelConfig::from_bytes(json).expect("parses");
        assert_eq!(c.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(c.num_hidden_layers, Some(32));
        assert_eq!(c.num_key_value_heads, Some(8));
        assert_eq!(c.summary(), "LlamaForCausalLM (32 layers, hidden=4096)");
    }

    #[test]
    fn missing_fields_are_none() {
        let json = br#"{"architectures": ["Foo"]}"#;
        let c = ModelConfig::from_bytes(json).expect("parses");
        assert_eq!(c.architectures, vec!["Foo"]);
        assert_eq!(c.num_hidden_layers, None);
    }

    #[test]
    fn parses_safetensors_index() {
        let json = br#"{
            "metadata": {"total_size": 16060522496},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
                "model.layers.31.mlp.down_proj.weight": "model-00004-of-00004.safetensors"
            }
        }"#;
        let i = SafetensorsIndex::from_bytes(json).expect("parses");
        assert_eq!(i.weight_map.len(), 3);
        assert!(i.weight_map.contains_key("model.embed_tokens.weight"));
        assert!(i
            .weight_map
            .contains_key("model.layers.31.mlp.down_proj.weight"));
    }

    #[test]
    fn bad_json_returns_none() {
        assert!(ModelConfig::from_bytes(b"not json").is_none());
        assert!(SafetensorsIndex::from_bytes(b"{").is_none());
    }
}
