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
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
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
    /// `points.bin`, `meta.json`). Open `index.html` over HTTP in a browser.
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
    /// 2–16384, default 1024). Higher is more detailed. Above `COARSE_CAP` (128)
    /// the up-front download stays small and fixed — a coarse `COARSE_CAP`³
    /// fallback plus a sparse octree — while the fine detail streams on demand
    /// from a brick pool as you pan/zoom. The octree page structure is
    /// O(occupied), so raising `N` costs more disk on the host but not a bigger
    /// download, client RAM, or VRAM. Ignored in 2D mode.
    #[arg(long, default_value_t = 1024)]
    pub(crate) grid: u32,

    /// Advanced 3D override (power of two, 8–8192): hand-tune the coarse/detail
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
    /// suffixed " diff" / " moe" — e.g. "arbvis" / "arbvis moe")
    #[arg(long, value_name = "TITLE")]
    pub(crate) title: Option<String>,

    /// Color regions by xorb ID for xet-backed files; hue = xorb, intensity = byte.
    #[arg(long)]
    pub(crate) show_xet_xorbs: bool,

    /// Tile output format. AVIF (default) is ~30-50% smaller than PNG over
    /// the wire; PNG is the universal fallback.
    #[arg(long, value_enum, default_value_t = TileFormatArg::Avif)]
    pub(crate) tile_format: TileFormatArg,

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
    /// `hf_url::resolve` + mmap and are unaffected by `--stream`.
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
                    // Local --out <dir>: always disk-backed.
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

    /// Coarse destination shape for [`SourceCtx`], hiding the tempdir / upload
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

/// Read `--files` and `--file-list` into a single flat path list.
pub(crate) fn collect_input_files(
    files: Vec<PathBuf>,
    file_list: Option<PathBuf>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = files;
    if let Some(list_path) = file_list {
        let reader: Box<dyn Read> = if list_path.as_os_str() == "-" {
            Box::new(io::stdin())
        } else {
            Box::new(
                File::open(&list_path)
                    .with_context(|| format!("failed to open {}", list_path.display()))?,
            )
        };
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                out.push(PathBuf::from(trimmed));
            }
        }
    }
    Ok(out)
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
    use super::validate_grid;

    #[test]
    fn grid_cap_raised_to_16384() {
        assert!(validate_grid(16384).is_ok());
        assert!(validate_grid(1024).is_ok()); // the default
        assert!(validate_grid(32768).is_err()); // above the cap
        assert!(validate_grid(768).is_err()); // not a power of two
    }
}
