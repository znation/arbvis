//! The byte-only CLI surface: parsed [`Args`], the tile-format choice, output
//! destination resolution, and flag validation / input collection helpers.
//!
//! Split out of `lib.rs` so the orchestration layer ([`crate::pipeline`]) and
//! the built-in source providers ([`crate::providers`]) each read as one
//! responsibility. Downstream specializations (e.g. modelweightvis) flatten
//! `Args` into their own clap struct and call `crate::run`.

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use tempfile::TempDir;

use crate::hf_url;
use crate::registry::DestKind;
use crate::TileFormat;

/// CLI tile-format choice. Maps to a `(leaf, pyramid)` pair of [`TileFormat`]s.
///
/// AVIF is the default: ~30-50% smaller than PNG and supported in every
/// modern browser. Pick `png` only for byte-for-byte regression checks or
/// for the rare audience without AVIF support.
#[derive(Clone, Copy, Debug, ValueEnum, Default, PartialEq, Eq)]
pub(crate) enum TileFormatArg {
    #[default]
    Avif,
    Png,
}

impl TileFormatArg {
    /// Returns `(leaf_format, pyramid_format)`. Leaf tiles are encoded
    /// near-lossless (each pixel is one source byte; users may inspect),
    /// pyramid tiles are lossy (averaged content tolerates a few QP steps).
    pub(crate) fn split(self) -> (TileFormat, TileFormat) {
        match self {
            TileFormatArg::Avif => (
                // Leaf speed 8 (was 6) to match the pyramid preset. Only bites
                // for Xet-mode leaves, which are the sole AVIF leaf tiles —
                // Plain/Dtype leaves are indexed PNG (see `TileFormat` docs). For
                // those AVIF leaves it trims rav1e's rate-distortion search
                // (`rdo_mode_decision`/`rdo_partition_decision`, the dominant cost
                // in the encode profile) for a modest size bump; at quality 100
                // the output stays near-lossless regardless of speed.
                TileFormat::Avif {
                    quality: 100,
                    speed: 8,
                },
                TileFormat::Avif {
                    quality: 85,
                    speed: 8,
                },
            ),
            TileFormatArg::Png => (TileFormat::Png, TileFormat::Png),
        }
    }
}

/// Visualize binary files as Hilbert curve plots.
///
/// Each byte is mapped to a color and placed along a Hilbert curve, so
/// structural patterns in the file (e.g. repeated null regions, ASCII text,
/// high-entropy compressed data) become visually apparent.
///
/// Reads from FILES if provided, otherwise reads from stdin.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Files to visualize (defaults to stdin); multiple files are concatenated
    #[arg(conflicts_with = "diff")]
    pub(crate) files: Vec<PathBuf>,

    /// Read file list from this file (one path per line), or - for stdin
    #[arg(short = 'l', long, conflicts_with = "diff")]
    pub(crate) file_list: Option<PathBuf>,

    /// Write the viewer bundle to this directory. In the default 2D mode this is
    /// a Leaflet tile pyramid (`tiles/`, `index.html`, `labels.json`); under
    /// `--3d` it is the Three.js volume bundle (`index.html`, `volume.bin`,
    /// `bricks.bin`, `pagetable.bin`, `meta.json`). Open `index.html` over
    /// HTTP in a browser.
    ///
    /// Accepts a local directory or an `hf://` URL to upload the bundle to a Hub
    /// repo. Note: `hf://` upload does NOT stand up a Space; the `index.html`
    /// lands in the target repo but won't render on the Hub on its own. Use
    /// `--space` for a working visualization URL.
    #[arg(short = 'o', long)]
    pub(crate) out: Option<PathBuf>,

    /// Render in 3D: lay bytes along a 3D Hilbert curve in a cube and emit a
    /// Three.js viewer (volume + point-cloud modes) instead of the 2D tile
    /// pyramid. Opacity encodes density so the cube's interior is visible.
    #[arg(long = "3d")]
    pub(crate) three_d: bool,

    /// 3D target detail resolution: the voxel grid side (a power of two,
    /// 2–16384, default 1024). Higher is more detailed. Above `COARSE_CAP` (256)
    /// the up-front download stays small and fixed — a coarse `COARSE_CAP`³
    /// fallback plus a sparse octree — while the fine detail streams on demand
    /// from a brick pool as you pan/zoom. The octree page structure is
    /// O(occupied), so raising `N` costs more disk on the host but not a bigger
    /// download, client RAM, or VRAM. Ignored in 2D mode.
    #[arg(long, default_value_t = 1024)]
    pub(crate) grid: u32,

    /// Advanced 3D override (power of two, 8–16384): hand-tune the coarse/detail
    /// split. When set, the dense coarse grid is built at `--grid` and the
    /// streamed brick pool at `--volume-res` (so the volume can exceed the
    /// coarse grid for sparse data). `0` (default) derives the split from
    /// `--grid` automatically (see `volume::derive_volume_resolution`). Byte
    /// path only; ignored in 2D.
    #[arg(long = "volume-res", default_value_t = 0)]
    pub(crate) volume_res: u32,

    /// Visualize abs(modified - original) byte differences; ORIGINAL and MODIFIED are files or directories
    #[arg(long, num_args = 2, value_names = ["ORIGINAL", "MODIFIED"])]
    pub(crate) diff: Option<Vec<PathBuf>>,

    // Specialization-specific flags (a multi-scene view, a structured layout
    // choice, format-specific diff knobs, etc.) live on the *downstream*
    // crate's clap struct, which flattens this one. The downstream reads those
    // flags to construct its own `SourceProvider`s / set `Registry::layout_mode`
    // before calling `run`; arbvis-only callers never see them in `arbvis --help`.
    /// Render and deploy a viewable HF Space (e.g. username/my-vis). Creates the
    /// Space with a Docker app that serves the viewer, and stores the bundle in a
    /// sibling bucket auto-named `<namespace>/<repo>_bucket`. Works for both 2D
    /// and `--3d`.
    ///
    /// Contrast with `--out hf://...`, which uploads only the viewer bundle
    /// (no Space scaffolding). Combine with `--out <local_dir>` and no input
    /// files to re-deploy an already-rendered bundle without re-rendering.
    #[arg(long)]
    pub(crate) space: Option<String>,

    /// Regenerate index.html for an existing bundle directory without re-rendering
    #[arg(long, value_name = "DIR", conflicts_with_all = ["files", "diff", "out", "space"])]
    pub(crate) regen_html: Option<PathBuf>,

    /// Title shown in the HTML info panel (default: the brand name, optionally
    /// suffixed " diff" / " moe" — e.g. "arbvis" / "arbvis moe"). With
    /// `--regen-html`, an explicit title overrides the one in the bundle
    /// (2D bundles fall back to the brand name; 3D bundles re-read the title
    /// persisted in `meta.json`).
    #[arg(long, value_name = "TITLE")]
    pub(crate) title: Option<String>,

    /// Color regions by xorb ID for xet-backed files; hue = xorb, intensity = byte.
    #[arg(long)]
    pub(crate) show_xet_xorbs: bool,

    /// Write the entire 2D render as one PNG file (plain mode: indexed, one
    /// pixel per byte, same byte-color scheme as the tile pyramid; diff mode:
    /// truecolor with the signed-delta LUT, crosshatch fills, and tints;
    /// `--show-xet-xorbs`: truecolor xorb coloring, Tableau hue scaled by
    /// byte) instead of a viewer bundle — for embedding arbvis output in docs and
    /// PRs without serving a web bundle. With `--out DIR`, FILE is placed
    /// inside DIR; otherwise FILE is used as given.
    #[arg(
        long = "png",
        value_name = "FILE",
        conflicts_with_all = ["three_d", "space", "regen_html"]
    )]
    pub(crate) png: Option<PathBuf>,

    /// Tile output format. AVIF (default) is ~30-50% smaller than PNG over
    /// the wire; PNG is the universal fallback. Only consumed by the 2D
    /// tiled viewer; the 3D volume path encodes its own brick textures, so
    /// an explicitly-set value there is reported as ignored.
    /// `None` means the flag was not passed (the default applies).
    #[arg(long, value_enum)]
    pub(crate) tile_format: Option<TileFormatArg>,

    /// Opt in to streaming I/O. Keeps `hf://` inputs remote (per-tile range
    /// fetches instead of an up-front download) and — when combined with an
    /// `hf://` output or `--space` — pushes tiles to the Hub as they're
    /// produced rather than staging through a local tempdir. Off by default;
    /// the disk-backed path is much faster and more recoverable. Use
    /// `--stream` when input or output data won't fit on local disk.
    ///
    /// Applies to every flow that takes `hf://` inputs: the normal renderer,
    /// `--diff` (when both sides are repo-level URLs), and the MoE viewer
    /// (`--moe`).
    /// Single-file / local-path inputs always resolve through
    /// `hf_url::resolve` + snapshot and are unaffected by `--stream`.
    #[arg(long)]
    pub(crate) stream: bool,
}

/// Where the render output goes after rendering. Owns any temporary
/// directories so they live until the upload step completes.
///
/// `_tempdir: Option<TempDir>` is a **named** binding-with-leading-underscore,
/// NOT a wildcard pattern. The TempDir's `Drop` impl removes the directory
/// from disk; we need the binding alive until the post-render upload reads
/// from `local`. Rust drops named bindings at end-of-scope (in
/// `dispatch_render`'s match arms that is after the `.await` returns), which
/// is what keeps the upload reading a real path. If a future refactor renames
/// `_tempdir` to plain `_`, that becomes a wildcard pattern that drops
/// IMMEDIATELY at the destructure point — the directory is gone before the
/// upload starts. Do not rename.
pub(crate) enum OutputDest {
    /// `--out <dir>` and/or `--space`: a web-viewer bundle directory (a 2D tile
    /// pyramid, or — under `--3d` — a volume bundle).
    ///
    /// `local` is the disk path the bundle renders into (a user dir, or a
    /// tempdir inside `_tempdir`). It is `None` only for the 2D streaming path
    /// (`--stream` + HF destination), which pushes tiles as produced and never
    /// touches local disk; 3D always stages locally.
    Bundle {
        local: Option<PathBuf>,
        upload_hf: Option<String>,
        space: Option<String>,
        _tempdir: Option<TempDir>,
    },
}

impl OutputDest {
    /// Resolve the user's `--out`/`--space` flags into the bundle destination.
    ///
    /// Tempdirs are allocated lazily: only when the disk-backed path will
    /// actually use one (`hf://` output without `--stream`, or any `--3d` run,
    /// which always stages locally). With 2D `--stream`, streaming destinations
    /// skip the tempdir entirely so a read-only or full `/tmp` doesn't kill the
    /// run before it starts — which is exactly the environment `--stream` exists for.
    pub(crate) fn from_args(args: &Args) -> anyhow::Result<Self> {
        let out_set = args.out.is_some();
        let space_set = args.space.is_some();
        if !out_set && !space_set {
            anyhow::bail!(
                "no output specified: pass --out <DIR> to write the viewer bundle \
                 (a local dir or an hf:// URL), or --space <NAMESPACE/REPO> to \
                 render and deploy a Hugging Face Space."
            );
        }

        // Reject `--out hf://X --space S`: the two flags are documented
        // alternatives (see `--space` help: "Contrast with --out hf://..."),
        // and silently picking one over the other would mean the same flags
        // produce different end-states. `--space` + `--out <local_dir>` is fine
        // and used by the deploy-only shortcut for re-deploys.
        if let (Some(p), true) = (args.out.as_ref(), space_set) {
            if hf_url::is_hf_path(p) {
                anyhow::bail!(
                    "--out hf://… and --space are alternatives, not stackable: \
                     --space deploys via its own bucket; --out hf:// uploads to \
                     a separate repo. Pass one, or combine --space with --out \
                     <local_dir> to (re-)deploy an already-rendered bundle."
                );
            }
        }

        // 3D always stages to a local dir (no streaming output path); 2D skips
        // staging only under --stream to an HF destination.
        let needs_staging = !args.stream || args.three_d;

        let (local, upload_hf, tempdir) = match &args.out {
            Some(p) => {
                if hf_url::is_hf_path(p) {
                    let url = p.to_string_lossy().into_owned();
                    if needs_staging {
                        let td = tempfile::tempdir().context("creating output tempdir")?;
                        (Some(td.path().to_path_buf()), Some(url), Some(td))
                    } else {
                        (None, Some(url), None)
                    }
                } else {
                    // Local --out <dir>: always disk-backed. Fail before the
                    // (possibly long) render if the path exists but is not a
                    // directory — otherwise the failure surfaces only at
                    // final tile write, as an opaque os error with no path.
                    if p.exists() && !p.is_dir() {
                        anyhow::bail!(
                            "--out {} is a file, not a directory; pass a directory \
                             path to write the viewer bundle into",
                            p.display()
                        );
                    }
                    (Some(p.clone()), None, None)
                }
            }
            None => {
                // --space without --out: tempdir for disk-backed sync;
                // nothing for streaming (uploads go through the bucket sink).
                if needs_staging {
                    let td = tempfile::tempdir().context("creating output tempdir")?;
                    (Some(td.path().to_path_buf()), None, Some(td))
                } else {
                    (None, None, None)
                }
            }
        };

        Ok(OutputDest::Bundle {
            local,
            upload_hf,
            space: args.space.clone(),
            _tempdir: tempdir,
        })
    }

    /// Coarse destination shape for [`crate::registry::SourceCtx`], hiding the tempdir / upload
    /// internals so a provider can gate on it without depending on them.
    pub(crate) fn kind(&self) -> DestKind {
        match self {
            OutputDest::Bundle { .. } => DestKind::Bundle,
        }
    }
}

/// Validate `--grid`: a power of two in `[2, 16384]`. Detail above the coarse cap
/// streams (see `volume::derive_volume_resolution`) from a sparse octree whose
/// page structure is O(occupied) — so neither the download, the client RAM, nor
/// the VRAM scales with the volume, only with the data actually present. The
/// upper bound caps the Hilbert order (16384 = order 14).
pub(crate) fn validate_grid(side: u32) -> anyhow::Result<()> {
    if !(2..=16384).contains(&side) || !side.is_power_of_two() {
        anyhow::bail!("--grid must be a power of two between 2 and 16384, got {side}");
    }
    Ok(())
}

/// Validate `--volume-res`: `0` (derive from `--grid`) or a power of two in
/// `[8, 16384]`. The sparse brick pool + octree store only occupied data, so a
/// high resolution is affordable; the upper bound caps the Hilbert order (16384
/// = order 14).
pub(crate) fn validate_volume_res(res: u32) -> anyhow::Result<()> {
    if res != 0 && (!(8..=16384).contains(&res) || !res.is_power_of_two()) {
        anyhow::bail!("--volume-res must be 0 or a power of two between 8 and 16384, got {res}");
    }
    Ok(())
}

/// Warning text for `--volume-res` set to a value that cannot stream: the
/// sparse brick pool only exists when `--volume-res` is strictly above
/// `--grid` (see `aggregate`'s `order_v` gate), so an at-or-below value builds
/// a dense grid at `--grid` and silently drops the streamed detail the flag
/// asked for. Returns `None` when the setting is usable (0, or above `--grid`).
pub(crate) fn volume_res_ignored_warning(grid: u32, volume_res: u32) -> Option<String> {
    if volume_res == 0 || volume_res > grid {
        return None;
    }
    Some(format!(
        "--volume-res {volume_res} is not above --grid {grid}; the streamed brick pool only \
         exists when --volume-res > --grid, so this run builds a dense grid at --grid with no \
         streamed detail. Pass a power of two above {grid}, or 0 to derive the split automatically."
    ))
}

/// Names of 3D-only flags whose values differ from their defaults and would
/// therefore be silently ignored without `--3d`.
pub(crate) fn ignored_3d_flags(grid: u32, volume_res: u32) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if grid != 1024 {
        flags.push("--grid");
    }
    if volume_res != 0 {
        flags.push("--volume-res");
    }
    flags
}

/// Warning text for 2D runs that set 3D-only flags, with the verb and pronoun
/// agreeing with how many flags are actually set.
pub(crate) fn ignored_3d_flags_warning(ignored: &[&str]) -> String {
    if ignored.len() == 1 {
        format!(
            "{} only takes effect with --3d; ignoring it in 2D mode",
            ignored[0]
        )
    } else {
        format!(
            "{} only take effect with --3d; ignoring them in 2D mode",
            ignored.join(", ")
        )
    }
}

/// Pick the viewer title: the user's `--title` if set, else the brand name
/// with a mode suffix (`"{name} moe"` / `"{name} diff"`, or just `"{name}"`
/// when `suffix` is empty). Built once per run, so the fallback allocation is
/// negligible.
pub(crate) fn default_title(user: Option<String>, name: &str, suffix: &str) -> Cow<'static, str> {
    match user {
        Some(t) => Cow::Owned(t),
        None if suffix.is_empty() => Cow::Owned(name.to_string()),
        None => Cow::Owned(format!("{name} {suffix}")),
    }
}

/// Fail when a run has no input files *and* stdin is a terminal.
///
/// With no positional FILES and no `--file-list`, the byte provider falls back
/// to reading stdin to EOF. On an interactive terminal that reads forever —
/// the user just sees a hang — so fail with usage guidance instead. A
/// non-terminal stdin (pipe, file redirect) still falls through to the
/// provider, so `arbvis < file.bin` keeps working.
///
/// Callers pass `stdin_is_terminal` rather than checking here so this stays
/// unit-testable without a TTY.
pub(crate) fn check_bare_run_inputs(
    files_empty: bool,
    stdin_is_terminal: bool,
) -> anyhow::Result<()> {
    if files_empty && stdin_is_terminal {
        anyhow::bail!(
            "no input files given and stdin is a terminal — pass file paths as arguments, \
             use --file-list, or pipe data on stdin (e.g. `arbvis < file.bin`)"
        );
    }
    Ok(())
}

/// Read `--files` and `--file-list` into a single flat path list.
pub(crate) fn collect_input_files(
    files: Vec<PathBuf>,
    file_list: Option<PathBuf>,
    stdin_is_terminal: bool,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = files;
    if let Some(list_path) = file_list {
        let reader: Box<dyn Read> = if list_path.as_os_str() == "-" {
            // `-` means "read the list from stdin"; on an interactive terminal
            // that blocks forever (same hang as the bare-run stdin fallback,
            // but *inside* this function — the later `check_bare_run_inputs`
            // guard never gets a chance to fire). Fail with usage guidance
            // instead; piped/redirected stdin still falls through.
            if stdin_is_terminal {
                anyhow::bail!(
                    "--file-list - reads the file list from stdin, which is a terminal — \
                     pipe the list on stdin (e.g. `ls *.bin | arbvis --file-list - --out dir`), \
                     or pass a list file path"
                );
            }
            Box::new(io::stdin())
        } else {
            Box::new(
                File::open(&list_path)
                    .with_context(|| format!("failed to open {}", list_path.display()))?,
            )
        };
        let mut listed = 0usize;
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                out.push(PathBuf::from(trimmed));
                listed += 1;
            }
        }
        if listed == 0 {
            // An empty *input list* means stdin, but an empty *file list* is
            // a user mistake (typically a typo'd --file-list path resolved by
            // shell glob, or a list filtered down to nothing). Fail fast
            // instead of silently visualizing whatever stdin holds — or
            // blocking forever on an interactive terminal.
            anyhow::bail!(
                "--file-list {}: contains no paths (blank lines are ignored)",
                list_path.display()
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod file_list_tests {
    use super::{collect_input_files, Args, OutputDest};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn file_list_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(&list, "a.bin\n\n  \nb.bin\n").unwrap();
        let files = collect_input_files(vec![], Some(list), false).unwrap();
        assert_eq!(files, vec![PathBuf::from("a.bin"), PathBuf::from("b.bin")]);
    }

    #[test]
    fn empty_file_list_fails_instead_of_reading_stdin() {
        let dir = tempfile::tempdir().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(&list, "\n   \n").unwrap();
        let err = collect_input_files(vec![], Some(list), false).unwrap_err();
        assert!(err.to_string().contains("contains no paths"), "{err}");
    }

    #[test]
    fn file_list_stdin_on_terminal_fails_instead_of_hanging() {
        let err = collect_input_files(vec![], Some(PathBuf::from("-")), true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--file-list -"), "{msg}");
        assert!(msg.contains("terminal"), "{msg}");
        // Non-terminal stdin (the piped case) must still fall through to the
        // read — verified in file_list_skips_blank_lines for a file path; here
        // only the guard itself is asserted, since a unit test cannot safely
        // block on a real terminal stdin.
    }

    #[test]
    fn out_pointing_at_a_regular_file_fails_fast() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("afile");
        std::fs::write(&file, b"x").unwrap();
        let args = Args::try_parse_from([
            "arbvis",
            file.to_str().unwrap(),
            "--out",
            file.to_str().unwrap(),
        ])
        .unwrap();
        let err = match OutputDest::from_args(&args) {
            Err(e) => e,
            Ok(_) => panic!("--out pointing at a regular file should fail"),
        };
        let msg = err.to_string();
        assert!(msg.contains("is a file, not a directory"), "{msg}");
        assert!(msg.contains("afile"), "{msg}");
        // A path that does not exist yet (the common case) must still be
        // accepted — creation happens later, at render time.
        let args = Args::try_parse_from([
            "arbvis",
            file.to_str().unwrap(),
            "--out",
            dir.path().join("newdir").to_str().unwrap(),
        ])
        .unwrap();
        assert!(OutputDest::from_args(&args).is_ok());
    }

    #[test]
    fn dash_means_stdin_and_no_list_leaves_files_alone() {
        // No --file-list: the list is passed through untouched (stdin fallback
        // happens downstream in prepare_sources, not here).
        let files = collect_input_files(vec!["x.bin".into()], None, false).unwrap();
        assert_eq!(files, vec![PathBuf::from("x.bin")]);
        // `-` reads stdin directly, so it is exercised only in integration use,
        // not in unit tests (reading a live stdin would block the suite).
    }
}

#[cfg(test)]
mod title_tests {
    use super::default_title;

    #[test]
    fn brand_and_suffix_fallbacks() {
        assert_eq!(
            default_title(None, "modelweightvis", "moe"),
            "modelweightvis moe"
        );
        assert_eq!(default_title(None, "arbvis", "diff"), "arbvis diff");
        assert_eq!(default_title(None, "arbvis", ""), "arbvis");
    }

    #[test]
    fn user_title_wins() {
        assert_eq!(
            default_title(Some("custom".to_string()), "arbvis", "moe"),
            "custom"
        );
    }
}

#[cfg(test)]
mod grid_validation_tests {
    use super::{
        ignored_3d_flags, ignored_3d_flags_warning, validate_grid, validate_volume_res,
        volume_res_ignored_warning,
    };

    #[test]
    fn grid_cap_raised_to_16384() {
        assert!(validate_grid(16384).is_ok());
        assert!(validate_grid(1024).is_ok()); // the default
        assert!(validate_grid(32768).is_err()); // above the cap
        assert!(validate_grid(768).is_err()); // not a power of two
    }

    #[test]
    fn volume_res_cap_raised_to_16384() {
        assert!(validate_volume_res(0).is_ok()); // derive from --grid
        assert!(validate_volume_res(16384).is_ok());
        assert!(validate_volume_res(8192).is_ok());
        assert!(validate_volume_res(32768).is_err()); // above the cap
        assert!(validate_volume_res(1000).is_err()); // not a power of two
    }

    #[test]
    fn volume_res_ignored_warning_only_fires_at_or_below_grid() {
        assert!(volume_res_ignored_warning(1024, 0).is_none()); // derive
        assert!(volume_res_ignored_warning(1024, 2048).is_none()); // streams
        let msg = volume_res_ignored_warning(1024, 512).expect("below grid should warn");
        assert!(msg.contains("--volume-res 512"));
        assert!(msg.contains("--grid 1024"));
        assert!(volume_res_ignored_warning(1024, 1024)
            .expect("equal to grid cannot stream either")
            .contains("no streamed detail"));
    }

    #[test]
    fn ignored_3d_flags_only_names_non_defaults() {
        assert!(ignored_3d_flags(1024, 0).is_empty()); // both defaults
        assert_eq!(ignored_3d_flags(512, 0), vec!["--grid"]);
        assert_eq!(ignored_3d_flags(1024, 4096), vec!["--volume-res"]);
        assert_eq!(ignored_3d_flags(512, 4096), vec!["--grid", "--volume-res"]);
    }

    #[test]
    fn ignored_3d_flags_warning_agrees_with_flag_count() {
        assert_eq!(
            ignored_3d_flags_warning(&["--grid"]),
            "--grid only takes effect with --3d; ignoring it in 2D mode"
        );
        assert_eq!(
            ignored_3d_flags_warning(&["--grid", "--volume-res"]),
            "--grid, --volume-res only take effect with --3d; ignoring them in 2D mode"
        );
    }
}

#[cfg(test)]
mod png_flag_tests {
    use super::{check_bare_run_inputs, Args, OutputDest, TileFormatArg};
    use clap::Parser;

    #[test]
    fn tile_format_none_by_default_and_some_when_passed() {
        let default = Args::parse_from(["arbvis", "a.bin"]);
        assert_eq!(default.tile_format, None);
        let explicit = Args::parse_from(["arbvis", "a.bin", "--tile-format", "png"]);
        assert_eq!(explicit.tile_format, Some(TileFormatArg::Png));
    }

    #[test]
    fn png_conflicts_with_3d_space_regen_and_composes_with_xorbs() {
        let png = ["arbvis", "--png", "out.png"];
        assert!(Args::try_parse_from(png).is_ok());
        for flag in [
            vec!["--3d"],
            vec!["--space", "me/vis"],
            vec!["--regen-html", "dir"],
        ] {
            let mut argv = png.clone().to_vec();
            argv.extend_from_slice(&flag);
            assert!(
                Args::try_parse_from(&argv).is_err(),
                "--png should conflict with {}",
                flag.join(" ")
            );
        }
        // `--png` composes with `--diff` and `--show-xet-xorbs` (both export
        // as a single PNG).
        let with_diff = ["arbvis", "--diff", "a", "b", "--png", "out.png"];
        assert!(Args::try_parse_from(with_diff).is_ok());
        let with_xorbs = ["arbvis", "--show-xet-xorbs", "a.bin", "--png", "out.png"];
        assert!(Args::try_parse_from(with_xorbs).is_ok());
    }

    #[test]
    fn png_coexists_with_out_and_tile_format() {
        assert!(Args::try_parse_from([
            "arbvis",
            "--png",
            "out.png",
            "--out",
            "dir",
            "--tile-format",
            "png"
        ])
        .is_ok());
    }

    #[test]
    fn no_output_flags_fails_with_a_hint() {
        let args = Args::try_parse_from(["arbvis", "a.bin"]).unwrap();
        let err = match OutputDest::from_args(&args) {
            Err(e) => e,
            Ok(_) => panic!("no --out/--space should fail"),
        };
        let msg = err.to_string();
        assert!(msg.contains("no output specified"), "{msg}");
        assert!(msg.contains("--out <DIR>"), "{msg}");
        assert!(msg.contains("--space <NAMESPACE/REPO>"), "{msg}");
    }

    #[test]
    fn out_hf_url_plus_space_are_alternatives_not_stackable() {
        let args = Args::try_parse_from([
            "arbvis",
            "a.bin",
            "--out",
            "hf://org/repo",
            "--space",
            "org/space",
        ])
        .unwrap();
        let err = match OutputDest::from_args(&args) {
            Err(e) => e,
            Ok(_) => panic!("hf out + space should fail"),
        };
        let msg = err.to_string();
        assert!(msg.contains("alternatives, not stackable"), "{msg}");
    }

    #[test]
    fn space_plus_local_out_is_allowed() {
        let args = Args::try_parse_from([
            "arbvis",
            "a.bin",
            "--out",
            "outdir",
            "--space",
            "org/space",
        ])
        .unwrap();
        let dest = match OutputDest::from_args(&args) { Ok(d) => d, Err(e) => panic!("local out + space is legal: {e}") };
        let OutputDest::Bundle {
            local,
            upload_hf,
            space,
            ..
        } = &dest;
        assert_eq!(local.as_deref(), Some(std::path::Path::new("outdir")));
        assert_eq!(upload_hf.as_deref(), None);
        assert_eq!(space.as_deref(), Some("org/space"));
        assert_eq!(dest.kind(), crate::registry::DestKind::Bundle);
    }

    #[test]
    fn out_hf_streaming_skips_local_staging() {
        // 2D + --stream + hf:// out: tiles are pushed as produced, so no
        // tempdir is allocated (the point of --stream: a full /tmp must not
        // kill the run).
        let args = Args::try_parse_from([
            "arbvis",
            "a.bin",
            "--stream",
            "--out",
            "hf://org/repo",
        ])
        .unwrap();
        let dest = match OutputDest::from_args(&args) { Ok(d) => d, Err(e) => panic!("streaming hf out should be legal: {e}") };
        let OutputDest::Bundle {
            local, upload_hf, ..
        } = &dest;
        assert_eq!(local.as_deref(), None, "streaming must not stage locally");
        assert_eq!(upload_hf.as_deref(), Some("hf://org/repo"));
    }

    #[test]
    fn out_hf_disk_backed_stages_in_a_tempdir() {
        // Same flags without --stream: disk-backed upload needs a local
        // staging dir, allocated lazily as a tempdir.
        let args = Args::try_parse_from(["arbvis", "a.bin", "--out", "hf://org/repo"]).unwrap();
        let dest = match OutputDest::from_args(&args) { Ok(d) => d, Err(e) => panic!("disk-backed hf out should be legal: {e}") };
        let OutputDest::Bundle {
            local, upload_hf, ..
        } = &dest;
        let local = local.as_ref().expect("staging dir must exist");
        assert!(local.is_absolute(), "tempdir path {local:?} should be absolute");
        assert_eq!(upload_hf.as_deref(), Some("hf://org/repo"));
    }

    #[test]
    fn space_without_out_stages_locally_unless_streaming() {
        // --space alone: 3D-off but not streaming → disk-backed sync tempdir.
        let args =
            Args::try_parse_from(["arbvis", "a.bin", "--space", "org/space"]).unwrap();
        let dest = match OutputDest::from_args(&args) { Ok(d) => d, Err(e) => panic!("space-only should be legal: {e}") };
        let OutputDest::Bundle { local, space, .. } = &dest;
        assert!(local.as_ref().is_some(), "disk-backed space sync stages locally");
        assert_eq!(space.as_deref(), Some("org/space"));

        // 2D --stream --space: uploads go through the bucket sink, no tempdir.
        let args = Args::try_parse_from([
            "arbvis",
            "a.bin",
            "--stream",
            "--space",
            "org/space",
        ])
        .unwrap();
        let dest = match OutputDest::from_args(&args) { Ok(d) => d, Err(e) => panic!("streaming space-only should be legal: {e}") };
        let OutputDest::Bundle { local, space, .. } = &dest;
        assert_eq!(local.as_deref(), None, "streaming space sync must not stage locally");
        assert_eq!(space.as_deref(), Some("org/space"));
    }

    #[test]
    fn bare_run_input_check_fails_only_when_both_empty_and_tty() {
        // Non-terminal stdin: the documented `arbvis < file.bin` path — never errors.
        assert!(check_bare_run_inputs(true, false).is_ok());
        // Files present: always fine, terminal or not.
        assert!(check_bare_run_inputs(false, true).is_ok());
        assert!(check_bare_run_inputs(false, false).is_ok());
        // No files + interactive terminal: would block on stdin forever.
        let err = check_bare_run_inputs(true, true).expect_err("should fail");
        let msg = err.to_string();
        assert!(msg.contains("no input files"), "{msg}");
        assert!(msg.contains("stdin is a terminal"), "{msg}");
        assert!(msg.contains("--file-list"), "{msg}");
        assert!(msg.contains("arbvis < file.bin"), "{msg}");
    }
}
