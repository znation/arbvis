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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_part_after_full_file_name() {
        // The `.part` suffix attaches to the whole file name, replacing any
        // extension, so `bricks.bin` never stages as `bricks.part`.
        assert_eq!(part_path(Path::new("out/bricks.bin")), PathBuf::from("out/bricks.bin.part"));
        assert_eq!(part_path(Path::new("viewer.html")), PathBuf::from("viewer.html.part"));
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
}
