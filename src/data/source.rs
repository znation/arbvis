//! Local source construction and loading: `prepare_sources` (local paths and
//! stdin, with directory expansion) and `load_source_data` (turn a [`Source`]
//! into a fetchable [`Data`]).
//!
//! The type definitions these functions operate on live in [`super`]; the
//! mixed local/remote spec path and the Hub download plumbing live in
//! [`super::remote`], the diff builders in [`super::diff`].

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use indicatif::ProgressBar;
use memmap2::Mmap;

use crate::progress::{counter_style, multi};

use super::{Data, Extensions, Source, SourceKind};

/// Recursively collect the files under `root` as sorted, full `PathBuf`s.
/// Unreadable directories and unreadable entries are logged (warned) and
/// skipped.
pub fn collect_files_recursive(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files);
    files.sort();
    files
}

/// Recursively collects regular files under `dir`, following symlinks to
/// files and directories. Paths that exist but are neither a readable
/// regular file nor a directory — a dangling symlink left by an interrupted
/// download or checkout, a FIFO/socket, an entry the OS refuses to stat —
/// are warned about rather than vanishing silently: a caller building a
/// diff would otherwise treat the missing side as empty and render a wrong
/// answer with no explanation.
fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("{}: {} — skipping", dir.display(), e);
            return;
        }
    };
    for entry in entries {
        // One unreadable entry must not silently swallow its neighbors
        // (entries.flatten() dropped the whole entry with no warning).
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                log::warn!("{}: {} — skipping", dir.display(), e);
                continue;
            }
        };
        let path = entry.path();
        // Cheap non-following classification first, so a directory is never
        // charged for a stat of its whole subtree.
        // `entry.file_type()` does not follow symlinks, so classification is
        // done through the path with `is_dir`/`is_file`, which do: a link to a
        // directory is recursed into, a link to a regular file is collected,
        // and a dangling link (or FIFO/socket) lands in the else branch below.
        if path.is_dir() {
            collect_recursive(&path, files);
        } else if path.is_file() {
            files.push(path);
        } else {
            log::warn!(
                "{}: not a readable regular file (dangling symlink or special file) — skipping",
                path.display()
            );
        }
    }
}

/// Build a one-shot progress bar attached to the global `MultiProgress` so
/// it interleaves cleanly with log output. Always returns `Some(...)`; the
/// non-TTY case is handled by the hidden draw target on the global multi.
/// `Option<ProgressBar>` is kept in the signature so existing call sites that
/// pattern-match it continue to compile.
pub(super) fn setup_progress(label: &str, total: u64) -> Option<ProgressBar> {
    let pb = multi()
        .add(ProgressBar::new(total))
        .with_style(counter_style())
        .with_message(label.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    Some(pb)
}

/// Build sources and return total byte count.
///
/// Files are opened lazily (one at a time) to avoid exhausting OS fd limits.
/// Stdin is buffered into memory upfront since its size is unknown.
///
/// For .safetensors files: the header is parsed and attached as ModelInfo
/// for dtype coloring. The file is kept as a single Source (one per file) so that
/// inter-tensor borders are not drawn and the Hilbert curve flows smoothly across
/// the whole file with color transitions only at tensor boundaries.
pub fn prepare_sources(
    files: &[PathBuf],
    registry: &crate::registry::Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
    if files.is_empty() {
        log::info!("Reading stdin...");
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                kind: SourceKind::Buffered(buf),
                byte_size: len,
                name_override: None,
                xet_terms: None,
                extensions: Extensions::default(),
            }],
            len,
        ));
    }

    // Expand any directory paths (e.g. from a repo-level hf:// download) into
    // their constituent files so they can be treated as individual sources.
    let expanded: Vec<PathBuf> = files
        .iter()
        .flat_map(|p| {
            if p.is_dir() {
                collect_files_recursive(p)
            } else {
                vec![p.clone()]
            }
        })
        .collect();

    let mut sources = Vec::new();
    let mut total = 0u64;
    for path in &expanded {
        let size = match std::fs::metadata(path) {
            Ok(m) => m.len(),
            Err(e) => {
                log::warn!("{}: {} — skipping", path.display(), e);
                continue;
            }
        };

        // Ask each registered `FormatPlugin` whether it recognizes this
        // path; the first that does gets to populate the source's
        // typed-extensions map (e.g. with `ModelInfo`). arbvis itself
        // knows nothing format-specific.
        let mut extensions = Extensions::default();
        for plugin in &registry.formats {
            if plugin.detects_path(path) {
                if let Err(e) = plugin.populate_local(path, size, &mut extensions) {
                    log::warn!(
                        "{}: format plugin `{}` failed: {e} — treating as plain binary",
                        path.display(),
                        plugin.id()
                    );
                }
                break;
            }
        }

        total += size;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::File(path.clone()),
            byte_size: size,
            name_override: None,
            xet_terms: None,
            extensions,
        });
    }
    if sources.is_empty() {
        // Every explicitly named input failed to stat (typically a typo'd
        // path, each already logged as a warning above). Fail fast instead of
        // rendering a silent empty viewer. An empty *input list* is stdin and
        // handled above; an empty *result* from a non-empty list is an error.
        anyhow::bail!(
            "no readable input files: {} (see the skip warnings above)",
            files
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok((sources, total))
}

/// Load a source's bytes for random access: mmaps file sources, clones buffered sources.
/// For diff sources, returns a LazyDiff that computes bytes on demand per tile.
/// For Http sources, returns a `Data::Http` handle that fetches byte ranges on demand.
pub fn load_source_data(s: &Source) -> anyhow::Result<Data> {
    match &s.kind {
        SourceKind::File(p) => {
            let f = File::open(p)?;
            Ok(Data::Mapped(unsafe { Mmap::map(&f) }?))
        }
        SourceKind::Buffered(v) => Ok(Data::Owned(v.clone())),
        SourceKind::Diff { original, modified } => {
            let f_o = File::open(original)?;
            let f_m = File::open(modified)?;
            let m_o = Arc::new(unsafe { Mmap::map(&f_o) }?);
            let m_m = Arc::new(unsafe { Mmap::map(&f_m) }?);
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let m_o = Arc::clone(&m_o);
                let m_m = Arc::clone(&m_m);
                Box::pin(async move {
                    // Zero-pad reads beyond either side's length so that
                    // same-name files with different sizes can share one diff
                    // source. The longer side's tail diffs against zero.
                    let read_padded = |m: &Mmap| -> Vec<u8> {
                        let s = start as usize;
                        let mlen = m.len();
                        let mut buf = vec![0u8; len];
                        if s < mlen {
                            let take = (mlen - s).min(len);
                            buf[..take].copy_from_slice(&m[s..s + take]);
                        }
                        buf
                    };
                    let a = read_padded(&m_o);
                    let b = read_padded(&m_m);
                    Ok(diff_bytes_to_color(&a, &b))
                })
            })))
        }
        SourceKind::Http(spec) => Ok(Data::Http {
            repo: spec.repo.clone(),
            filename: Arc::clone(&spec.filename),
            revision: Arc::clone(&spec.revision),
        }),
        SourceKind::UnmatchedRegion { .. } => Ok(Data::ZeroFill),
        SourceKind::RangeDiff {
            orig,
            mod_,
            orig_start,
            mod_start,
        } => {
            let orig = Arc::clone(orig);
            let mod_ = Arc::clone(mod_);
            let orig_start = *orig_start;
            let mod_start = *mod_start;
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let orig = Arc::clone(&orig);
                let mod_ = Arc::clone(&mod_);
                Box::pin(async move {
                    let a = orig.fetch_range(orig_start + start, len).await?;
                    let b = mod_.fetch_range(mod_start + start, len).await?;
                    Ok(diff_bytes_to_color(&a, &b))
                })
            })))
        }
        SourceKind::OneSidedRange { data, start, .. } => Ok(Data::OffsetSlice {
            inner: Arc::clone(data),
            base: *start,
        }),
        SourceKind::Custom(cs) => cs.open(),
    }
}

#[cfg(test)]
mod collect_recursive_tests {
    use super::collect_recursive;

    /// A dangling symlink inside an input/diff directory must be *warned
    /// about*, not silently dropped: a directory diff would otherwise render
    /// the missing side as an "only in modified" region with no explanation.
    #[test]
    fn dangling_symlink_is_skipped_with_real_and_linked_files_kept() {
        #[cfg(unix)]
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("real.bin"), b"abc").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/nested.bin"), b"def").unwrap();
        #[cfg(unix)]
        {
            symlink(dir.path().join("real.bin"), dir.path().join("link.bin")).unwrap();
            // A symlink to a subdirectory must be recursed into, not skipped.
            symlink(dir.path().join("sub"), dir.path().join("sublink")).unwrap();
            symlink(
                dir.path().join("no-such-target"),
                dir.path().join("dangling.bin"),
            )
            .unwrap();
        }

        let mut files = Vec::new();
        collect_recursive(dir.path(), &mut files);

        let names: Vec<String> = files
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str().map(String::from)))
            .collect();
        assert!(names.contains(&"real.bin".to_string()));
        assert!(names.contains(&"nested.bin".to_string()));
        #[cfg(unix)]
        {
            assert!(
                names.contains(&"link.bin".to_string()),
                "a symlink pointing at a regular file must still be collected"
            );
            assert!(
                names.contains(&"nested.bin".to_string())
                    && files.iter().any(|p| p.starts_with(dir.path().join("sublink"))),
                "a symlink pointing at a directory must be recursed into"
            );
            assert!(
                !names.contains(&"dangling.bin".to_string()),
                "a dangling symlink must be excluded (warned about, not silently dropped)"
            );
        }
    }
}

#[cfg(test)]
mod prepare_sources_tests {
    use super::prepare_sources;
    use crate::registry::Registry;
    use std::path::PathBuf;

    #[test]
    fn missing_paths_fail_instead_of_rendering_empty() {
        let registry = Registry::with_defaults();
        let err = match prepare_sources(
            &[
                PathBuf::from("no-such-file-1.bin"),
                PathBuf::from("no-such-file-2.bin"),
            ],
            &registry,
        ) {
            Err(e) => e,
            Ok(_) => panic!("all-missing input list should fail"),
        };
        assert!(
            err.to_string().contains("no readable input files"),
            "unexpected error: {err:#}"
        );
        assert!(err.to_string().contains("no-such-file-1.bin"));
    }

    #[test]
    fn one_readable_path_among_missing_still_renders() {
        let registry = Registry::with_defaults();
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.bin");
        std::fs::write(&good, b"hello").unwrap();
        let (sources, total) =
            prepare_sources(&[good, PathBuf::from("no-such-file.bin")], &registry).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(total, 5);
    }

    #[test]
    fn directory_input_expands_to_all_files_recursively() {
        let registry = Registry::with_defaults();
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(dir.path().join("a.bin"), [0u8; 3]).unwrap();
        std::fs::write(nested.join("b.bin"), [0u8; 4]).unwrap();
        std::fs::write(dir.path().join("ignored.txt"), [0u8; 100]).unwrap();
        let (sources, total) = prepare_sources(&[dir.path().to_path_buf()], &registry).unwrap();
        // Every file under the directory becomes its own source, regardless
        // of extension or nesting depth.
        assert_eq!(sources.len(), 3);
        assert_eq!(total, 107);
        // Sources preserve the recursive traversal order and get fresh
        // consecutive file_idx values.
        // sort() orders by full path, so "a.bin" < "ignored.txt" <
        // "nested/b.bin" (the nested directory's prefix sorts as "n").
        let sizes: Vec<u64> = sources.iter().map(|s| s.byte_size).collect();
        assert_eq!(sizes, [3, 100, 4]);
        assert_eq!(
            sources.iter().map(|s| s.file_idx).collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }
}

fn diff_bytes_to_color(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| {
            let delta = b as i16 - a as i16;
            let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
            if delta >= 0 {
                127u8 + brightness
            } else {
                127u8 - brightness
            }
        })
        .collect()
}

#[cfg(test)]
mod diff_bytes_to_color_tests {
    use super::diff_bytes_to_color;

    #[test]
    fn equal_bytes_stay_neutral_and_deltas_shift_signed() {
        // Identical bytes map to the neutral 127 gray.
        assert_eq!(diff_bytes_to_color(&[5, 200], &[5, 200]), [127, 127]);
        // Positive delta brightens, negative delta darkens, scaled by
        // magnitude: a full ±255 swing lands exactly 127±127.
        assert_eq!(diff_bytes_to_color(&[0], &[255]), [254]);
        assert_eq!(diff_bytes_to_color(&[255], &[0]), [0]);
        // Mid-magnitude delta: 128/255 of the way from 127 toward the pole.
        assert_eq!(diff_bytes_to_color(&[0], &[128]), [191]);
        assert_eq!(diff_bytes_to_color(&[128], &[0]), [63]);
    }
}
