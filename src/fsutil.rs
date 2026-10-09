//! Small filesystem staging helpers shared across modules.

use std::path::{Path, PathBuf};

/// Stage path for an artifact being written atomically: `bricks.bin` is
/// staged at `bricks.bin.part` and renamed into place when complete.
///
/// Callers: `volume::write_atomic` (+ `volume::aggregate`), the tile writer
/// in `tiled` (`write_tile_file`), and `tiled::html::write_viewer_pair`.
pub(crate) fn part_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new(""))
        .to_os_string();
    name.push(".part");
    path.with_file_name(name)
}

/// Seal a staged artifact: rename `<file>.part` over its final path. If the
/// rename fails (target path is a directory, cross-device move, permissions),
/// the staged file is removed so it never outlives the failure — a partial
/// `bricks.bin` or tile must not be read by the viewer, pushed by a later
/// `hf upload`, or mistaken for an in-progress staging file.
///
/// Callers: `volume::write_atomic`, `volume::aggregate` (both seal sites),
/// `tiled::write_tile_file`, and `tiled::html::write_viewer_pair`.
pub(crate) fn seal_part(part: &Path, final_path: &Path) -> anyhow::Result<()> {
    if let Err(e) = std::fs::rename(part, final_path) {
        // Best effort: don't leave a stale partial staging file behind.
        let _ = std::fs::remove_file(part);
        return Err(anyhow::Error::new(e).context(format!("sealing {}", final_path.display())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_part_after_full_file_name() {
        // The `.part` suffix attaches to the whole file name, replacing any
        // extension, so `bricks.bin` never stages as `bricks.part`.
        assert_eq!(
            part_path(Path::new("out/bricks.bin")),
            PathBuf::from("out/bricks.bin.part")
        );
        assert_eq!(
            part_path(Path::new("viewer.html")),
            PathBuf::from("viewer.html.part")
        );
    }

    #[test]
    fn keeps_parent_directory() {
        assert_eq!(
            part_path(Path::new("a/b/c/tiles.bin")),
            PathBuf::from("a/b/c/tiles.bin.part")
        );
    }

    #[test]
    fn file_name_falls_back_to_empty_for_root() {
        // Paths with no file name (e.g. "/") stage a dotfile next to them.
        assert_eq!(part_path(Path::new("/")), PathBuf::from("/.part"));
    }

    #[test]
    fn seal_part_renames_and_cleans_up_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let part = dir.path().join("bricks.bin.part");
        std::fs::write(&part, b"sealed").unwrap();
        let final_path = dir.path().join("bricks.bin");
        seal_part(&part, &final_path).unwrap();
        assert!(final_path.exists());
        assert!(!part.exists());

        // A failed rename (target is a directory) must not leave the staged
        // file behind for a later upload or rerun to mistake for in-progress.
        let part2 = dir.path().join("bricks2.bin.part");
        std::fs::write(&part2, b"staged").unwrap();
        let target_dir = dir.path().join("bricks2.bin");
        std::fs::create_dir(&target_dir).unwrap();
        let err = seal_part(&part2, &target_dir).unwrap_err();
        assert!(err.to_string().contains("sealing"));
        assert!(!part2.exists());
    }
}
