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
