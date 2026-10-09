//! Scene grouping for multi-scene tiled output: bucketing sources by their
//! [`SceneTag`], path-safe slug sanitization, and the [`SceneGroup`] peeled
//! off the input by [`partition_scenes`].

use std::collections::HashMap;

use crate::data::{SceneTag, Source};

/// One scene's worth of sources, peeled off `Vec<Source>` by [`partition_scenes`].
pub(super) struct SceneGroup {
    /// `Some(key)` → tiles go under `tiles/<key>/`; `None` → legacy lone scene.
    pub(super) key: Option<String>,
    pub(super) label: String,
    pub(super) order: u32,
    pub(super) sources: Vec<Source>,
    pub(super) total: u64,
}

/// Reduce a plugin-supplied scene key to a path-safe slug: the key ends up as
/// `tiles/<key>/` in both the on-disk output tree and the Hub repo path, so a
/// key holding `/`, `..`, or an absolute path would make the tiler create or
/// write outside the tile root (and outside the intended repo subdirectory).
/// Everything outside `A-Za-z0-9_-` collapses to `_`; a key that sanitizes to
/// nothing (or to `.`/`..` shapes) falls back to `scene`.
fn sanitize_scene_key(key: &str) -> String {
    let slug: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let slug = slug.trim_matches('_');
    if slug.is_empty() {
        "scene".to_string()
    } else {
        slug.to_string()
    }
}

/// Group sources into scenes by their [`SceneTag`]. With no tags present, the
/// whole input is one implicit default scene (`key: None`) carrying the
/// caller's original `total` — preserving the exact legacy single-pyramid path.
/// With tags present, sources are bucketed by their sanitized `key`
/// (first-seen order, then sorted by `order`); each scene's `total` is the sum
/// of its source sizes.
pub(super) fn partition_scenes(sources: Vec<Source>, total: u64) -> Vec<SceneGroup> {
    let any_tagged = sources
        .iter()
        .any(|s| s.extensions.get::<SceneTag>().is_some());
    if !any_tagged {
        return vec![SceneGroup {
            key: None,
            label: String::new(),
            order: 0,
            sources,
            total,
        }];
    }

    let mut key_order: Vec<String> = Vec::new();
    let mut buckets: HashMap<String, (String, u32, Vec<Source>)> = HashMap::new();
    for s in sources {
        let (key, label, order) = match s.extensions.get::<SceneTag>() {
            Some(t) => (sanitize_scene_key(&t.key), t.label.clone(), t.order),
            // Untagged source in an otherwise-tagged run: bucket it into a
            // sensible default scene rather than dropping it.
            None => ("main".to_string(), "Main".to_string(), u32::MAX),
        };
        let entry = buckets.entry(key.clone()).or_insert_with(|| {
            key_order.push(key.clone());
            (label, order, Vec::new())
        });
        entry.2.push(s);
    }

    let mut groups: Vec<SceneGroup> = key_order
        .into_iter()
        .map(|k| {
            let (label, order, srcs) = buckets.remove(&k).unwrap();
            let total = srcs.iter().map(|s| s.byte_size).sum();
            SceneGroup {
                key: Some(k),
                label,
                order,
                sources: srcs,
                total,
            }
        })
        .collect();
    groups.sort_by_key(|g| g.order);
    groups
}

#[cfg(test)]
mod tests {
    use super::partition_scenes;
    use crate::data::{Extensions, SceneTag, Source, SourceKind};

    fn src(byte_size: u64, scene: Option<(&str, u32)>) -> Source {
        let mut extensions = Extensions::default();
        if let Some((key, order)) = scene {
            extensions.insert(SceneTag {
                key: key.to_string(),
                label: key.to_string(),
                order,
            });
        }
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(Vec::new()),
            byte_size,
            name_override: Some("t".to_string()),
            xet_terms: None,
            extensions,
        }
    }

    #[test]
    fn untagged_sources_form_one_default_scene_with_original_total() {
        let groups = partition_scenes(vec![src(10, None), src(20, None)], 999);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].key.is_none());
        // The lone default scene keeps the caller's `total` verbatim so the
        // legacy single-pyramid path is byte-for-byte unchanged.
        assert_eq!(groups[0].total, 999);
        assert_eq!(groups[0].sources.len(), 2);
    }

    #[test]
    fn tagged_sources_split_into_ordered_scenes_with_summed_totals() {
        // Deliberately interleaved and out-of-order to exercise grouping + sort.
        let groups = partition_scenes(
            vec![
                src(3, Some(("cka", 1))),
                src(10, Some(("summary", 0))),
                src(5, Some(("cka", 1))),
                src(20, Some(("summary", 0))),
            ],
            0,
        );
        assert_eq!(groups.len(), 2);
        // Sorted by `order`: summary (0) before cka (1).
        assert_eq!(groups[0].key.as_deref(), Some("summary"));
        assert_eq!(groups[0].total, 30);
        assert_eq!(groups[0].sources.len(), 2);
        assert_eq!(groups[1].key.as_deref(), Some("cka"));
        assert_eq!(groups[1].total, 8);
        assert_eq!(groups[1].sources.len(), 2);
    }

    #[test]
    fn hostile_scene_keys_are_reduced_to_path_safe_slugs() {
        // The key is interpolated into `tiles/{key}/` for on-disk joins and Hub
        // repo paths, so traversal and separator characters must not survive.
        for (key, want) in [
            ("../../evil", "evil"),
            ("/abs/path", "abs_path"),
            ("a/b\\c", "a_b_c"),
            ("..", "scene"),
            (".hidden", "hidden"),
            ("", "scene"),
            ("summary", "summary"), // well-formed keys pass through untouched
        ] {
            let groups = partition_scenes(vec![src(1, Some((key, 0)))], 0);
            assert_eq!(groups.len(), 1);
            assert_eq!(groups[0].key.as_deref(), Some(want), "key {key:?}");
        }
    }

    #[test]
    fn distinct_keys_sanitizing_to_the_same_slug_share_one_scene() {
        // Grouping happens on the sanitized key so a run cannot end up with
        // two scenes writing into the same `tiles/<slug>/` directory.
        let groups = partition_scenes(
            vec![
                src(1, Some(("a/b", 0))),
                src(2, Some(("a_b", 0))),
                src(4, Some(("cka", 1))),
            ],
            0,
        );
        assert_eq!(groups.len(), 2);
        let merged: Option<&super::SceneGroup> =
            groups.iter().find(|g| g.key.as_deref() == Some("a_b"));
        assert_eq!(merged.unwrap().sources.len(), 2);
    }
}
