//! Diff-source construction: the built-in `DiffSourceBuilder` impls for the
//! file-pair diff cases, plus the directory-tree byte diff. Split out of
//! `data.rs` so the core `Data`/`Source` plumbing stays readable in one
//! sitting.
//!
//! Directory-pair diffs stay inline in `prepare_diff_sources` for now —
//! they'll move behind the trait when format detection migrates to
//! `modelweightvis` (step 12).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::data::{DiffFill, Extensions, Source, SourceKind, collect_files_recursive};


fn is_json_path(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()),
        Some("json") | Some("jsonl")
    )
}

/// JSON / JSONL structure-aware diff. Applies when both paths have a
/// `.json` or `.jsonl` extension.
pub struct JsonDiffBuilder;

#[async_trait::async_trait]
impl crate::registry::DiffSourceBuilder for JsonDiffBuilder {
    fn id(&self) -> &'static str {
        "json"
    }
    fn priority(&self) -> i32 {
        200
    }
    async fn try_build(
        &self,
        ctx: &crate::registry::DiffBuildCtx<'_>,
    ) -> anyhow::Result<Option<(Vec<Source>, u64)>> {
        if !(is_json_path(ctx.original) && is_json_path(ctx.modified)) {
            return Ok(None);
        }
        let out =
            crate::json_diff::build_json_diff_sources(ctx.original, ctx.modified, ctx.is_finetune)
                .await?;
        Ok(Some(out))
    }
}

// `TensorDiffBuilder` lives in `modelweightvis::diff` (step 12e). The
// arbvis default registry no longer wires it up.

/// Plain-byte diff: builds one `SourceKind::Diff` source over a same-sized
/// pair. The floor of the builder priority stack — applies whenever the two
/// files exist and have the same size, and bails with an error if sizes
/// differ (matching the original `prepare_diff_sources` contract).
pub struct PlainBytesDiffBuilder;

#[async_trait::async_trait]
impl crate::registry::DiffSourceBuilder for PlainBytesDiffBuilder {
    fn id(&self) -> &'static str {
        "plain-bytes"
    }
    fn priority(&self) -> i32 {
        0
    }
    async fn try_build(
        &self,
        ctx: &crate::registry::DiffBuildCtx<'_>,
    ) -> anyhow::Result<Option<(Vec<Source>, u64)>> {
        let size_o = std::fs::metadata(ctx.original)?.len();
        let size_m = std::fs::metadata(ctx.modified)?.len();
        if size_o != size_m {
            anyhow::bail!(
                "--diff: file sizes differ ({} bytes vs {} bytes): {} vs {}",
                size_o,
                size_m,
                ctx.original.display(),
                ctx.modified.display()
            );
        }
        let source = Source {
            file_idx: 0,
            kind: SourceKind::Diff {
                original: ctx.original.to_path_buf(),
                modified: ctx.modified.to_path_buf(),
            },
            byte_size: size_o,
            name_override: None,
            xet_terms: None,
            extensions: Extensions::default(),
        };
        Ok(Some((vec![source], size_o)))
    }
}

/// Build diff sources from two files or two directories.
///
/// For files: dispatched through `registry.diffs` by descending priority. The
/// `PlainBytesDiffBuilder` floor always builds for a same-sized pair, so
/// iteration terminates with a valid diff for any well-formed input. A
/// specialization can register a higher-priority builder (e.g. a format-aware
/// element diff).
///
/// For directories: every file is byte-diffed by relative path via
/// [`byte_directory_diff`]. A specialization that diffs some files itself (e.g.
/// tensor matching across shards) runs its own [`crate::SourceProvider`] and
/// calls `byte_directory_diff` directly with a `skip` predicate for the
/// remainder.
pub async fn prepare_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    registry: &crate::registry::Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_is_file = original.is_file();
    let mod_is_file = modified.is_file();
    let orig_is_dir = original.is_dir();
    let mod_is_dir = modified.is_dir();

    if orig_is_file && mod_is_file {
        let ctx = crate::registry::DiffBuildCtx {
            original,
            modified,
            is_finetune,
        };
        let mut sorted: Vec<&Arc<dyn crate::registry::DiffSourceBuilder>> =
            registry.diffs.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.priority()));
        for builder in &sorted {
            if let Some(out) = builder.try_build(&ctx).await? {
                return Ok(out);
            }
        }
        anyhow::bail!(
            "--diff: no registered builder handled the input pair ({} vs {})",
            original.display(),
            modified.display()
        );
    }

    if orig_is_dir && mod_is_dir {
        // arbvis byte-only diffs every file by relative path. A specialization
        // that handles some files itself runs its own provider and calls
        // `byte_directory_diff` with a `skip` predicate for the remainder.
        let (sources, total) = byte_directory_diff(original, modified, is_finetune, &|_| false)?;
        if sources.is_empty() {
            anyhow::bail!("--diff: no matching file pairs found between the two directories");
        }
        return Ok((sources, total));
    }

    anyhow::bail!(
        "--diff: both arguments must be files or both must be directories (got {} and {})",
        if orig_is_file {
            "file"
        } else if orig_is_dir {
            "directory"
        } else {
            "missing path"
        },
        if mod_is_file {
            "file"
        } else if mod_is_dir {
            "directory"
        } else {
            "missing path"
        }
    );
}

/// Byte-diff two directory trees, matching files by relative path. Same-size
/// pairs become a `SourceKind::Diff`; size-mismatched pairs are byte-diffed
/// with zero-padding; files present on only one side become crosshatched
/// `UnmatchedRegion`s so they stay visible.
///
/// `skip` excludes entries the caller handles itself (e.g. a specialization's
/// own format-aware diff over the same directory) — pass `&|_| false` to diff
/// every file. `is_finetune` tunes crosshatch semantics: original-only files
/// render grey (expected drops) rather than red, and mod-only files are warned
/// about and rendered green.
///
/// Returns a possibly-empty `(sources, total)`; the caller decides whether an
/// empty result is an error, so a specialization can append the non-skipped
/// remainder to a larger source list (offsetting `file_idx` as needed).
pub fn byte_directory_diff(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    skip: &dyn Fn(&Path) -> bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_files = collect_files_recursive(original);
    let mod_files = collect_files_recursive(modified);

    let mut sources = Vec::new();
    let mut total = 0u64;

    // Match by relative path. Same-size pairs become a byte diff; different-size
    // or single-side files become crosshatched unmatched regions so they remain
    // visible.
    let orig_fill_kind = if is_finetune {
        DiffFill::Grey
    } else {
        DiffFill::Red
    };
    let orig_map: HashMap<PathBuf, PathBuf> = orig_files
        .iter()
        .filter(|p| !skip(p))
        .filter_map(|p| {
            p.strip_prefix(original)
                .ok()
                .map(|rel| (rel.to_path_buf(), p.clone()))
        })
        .collect();
    let mod_map: HashMap<PathBuf, PathBuf> = mod_files
        .iter()
        .filter(|p| !skip(p))
        .filter_map(|p| {
            p.strip_prefix(modified)
                .ok()
                .map(|rel| (rel.to_path_buf(), p.clone()))
        })
        .collect();

    let mut mod_only_keys: Vec<&PathBuf> = mod_map
        .keys()
        .filter(|k| !orig_map.contains_key(*k))
        .collect();
    mod_only_keys.sort();
    if is_finetune && !mod_only_keys.is_empty() {
        let names: Vec<String> = mod_only_keys
            .iter()
            .map(|rel| modified.join(rel).display().to_string())
            .collect();
        log::warn!(
            "--diff --finetune: modified side has {} file(s) with no counterpart on the \
                 original/base side — rendering as green crosshatch: {}",
            names.len(),
            names.join(", ")
        );
    }

    let mut sorted_keys: Vec<&PathBuf> = orig_map.keys().collect();
    sorted_keys.sort();

    for rel in sorted_keys {
        let orig_abs = &orig_map[rel];
        let size_o = match std::fs::metadata(orig_abs) {
            Ok(m) => m.len(),
            Err(e) => {
                log::warn!("{}: {} — skipping", orig_abs.display(), e);
                continue;
            }
        };
        match mod_map.get(rel) {
            None => {
                if size_o == 0 {
                    continue;
                }
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::UnmatchedRegion {
                        fill: orig_fill_kind,
                    },
                    byte_size: size_o,
                    name_override: Some(format!("[only in original] {}", rel.display())),
                    xet_terms: None,
                    extensions: Extensions::default(),
                });
                total += size_o;
            }
            Some(mod_abs) => {
                let size_m = match std::fs::metadata(mod_abs) {
                    Ok(m) => m.len(),
                    Err(e) => {
                        log::warn!("{}: {} — skipping", mod_abs.display(), e);
                        continue;
                    }
                };
                if size_o != size_m {
                    if is_finetune {
                        log::warn!(
                            "--diff --finetune: size mismatch ({} vs {} bytes) for {} — \
                                 byte-diffing with zero-padding on the shorter side",
                            size_o,
                            size_m,
                            rel.display()
                        );
                    } else {
                        log::warn!(
                                "size mismatch ({} vs {} bytes) for {} — byte-diffing with zero-padding",
                                size_o, size_m, rel.display()
                            );
                    }
                }
                let max_size = size_o.max(size_m);
                if max_size == 0 {
                    continue;
                }
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Diff {
                        original: orig_abs.clone(),
                        modified: mod_abs.clone(),
                    },
                    byte_size: max_size,
                    name_override: None,
                    xet_terms: None,
                    extensions: Extensions::default(),
                });
                total += max_size;
            }
        }
    }

    // mod-only files (non-finetune case — finetune bailed earlier).
    for rel in &mod_only_keys {
        let mod_abs = &mod_map[*rel];
        let size_m = match std::fs::metadata(mod_abs) {
            Ok(m) => m.len(),
            Err(e) => {
                log::warn!("{}: {} — skipping", mod_abs.display(), e);
                continue;
            }
        };
        if size_m == 0 {
            continue;
        }
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion {
                fill: DiffFill::Green,
            },
            byte_size: size_m,
            name_override: Some(format!("[only in modified] {}", rel.display())),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size_m;
    }

    Ok((sources, total))
}


#[cfg(test)]
mod tests {
    use super::{byte_directory_diff, prepare_diff_sources};
    use crate::registry::Registry;
    use std::fs;
    use std::path::Path;

    /// Skipped files must not produce sources, and same-size pairs become
    /// byte diffs while one-side-only files become unmatched regions.
    #[test]
    fn skip_predicate_and_unmatched_files_behave() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        fs::create_dir_all(a.join("sub")).unwrap();
        fs::create_dir_all(b.join("sub")).unwrap();
        fs::write(a.join("same.bin"), b"abc").unwrap();
        fs::write(b.join("same.bin"), b"abd").unwrap();
        fs::write(a.join("only_orig.bin"), b"xy").unwrap();
        fs::write(a.join("skip_me.bin"), b"zz").unwrap();
        fs::write(b.join("skip_me.bin"), b"zz").unwrap();

        let (sources, total) =
            byte_directory_diff(&a, &b, false, &|p: &Path| {
                p.file_name().is_some_and(|n| n == "skip_me.bin")
            })
            .unwrap();
        let names: Vec<String> = sources.iter().map(|s| s.name()).collect();
        assert!(names.iter().any(|n| n == "same.bin"), "names: {names:?}");
        assert!(
            names.iter().any(|n| n.contains("only_orig.bin")),
            "unmatched original file should produce an UnmatchedRegion source: {names:?}"
        );
        assert!(
            names.iter().all(|n| !n.contains("skip_me.bin")),
            "skipped files must not appear: {names:?}"
        );
        assert_eq!(total, 3 + 2, "same-size diff plus unmatched bytes");
        assert_eq!(sources[0].file_idx, 0);
    }
    /// File pairs go through the builder cascade: two same-sized non-JSON
    /// files fall through to the plain-bytes floor builder.
    #[tokio::test]
    async fn file_pair_falls_through_to_plain_bytes_builder() {
        let dir = tempfile::tempdir().unwrap();
        let (o, m) = (dir.path().join("o.bin"), dir.path().join("m.bin"));
        fs::write(&o, b"1234").unwrap();
        fs::write(&m, b"5678").unwrap();
        let registry = Registry::with_defaults();
        let (sources, total) = prepare_diff_sources(&o, &m, false, &registry).await.unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(total, 4);
        assert!(matches!(sources[0].kind, crate::data::SourceKind::Diff { .. }));
    }

    /// Mixed file/directory inputs are rejected with a clear message.
    #[tokio::test]
    async fn mixed_file_and_directory_input_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.bin");
        fs::write(&file, b"x").unwrap();
        let registry = Registry::with_defaults();
        let err = match prepare_diff_sources(&file, dir.path(), false, &registry).await {
            Err(e) => e,
            Ok(_) => panic!("mixed file/directory input should fail"),
        };
        assert!(err.to_string().contains("files or both must be directories"));
    }
}
