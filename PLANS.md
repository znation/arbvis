# Plans

Planned features, written by the plan loop and implemented by the feature loop.
Each plan: goal, approach, files touched, acceptance criteria. Move finished plans to Done.

## Planned

_None yet._

## Done

### Single-image PNG export (`--png FILE`) — Done 2026-10-09

**Implemented 2026-10-09 by feature.** As planned, with two clarifications: in PNG mode `OutputDest::from_args` is skipped entirely (so `--png` needs no `--out`; a given `--out DIR` is the PNG's parent, created if missing, and `hf://` outs are rejected); and the indexed PNG uses the full 256-entry `build_pixel_lut()` as its palette, so pixel index = byte value.

**Found by plan loop 2026-10-09.** Verified 2026-10-09: no single-image export exists — `src/lib.rs` `Args` has only `--tile-format` (tiles), and `grep -rni 'single.image\|snapshot' src README.md` shows no PNG-export path.

**Goal:** add a `--png FILE` flag that writes the entire 2D render as one PNG file (one pixel per byte, same byte-color scheme) instead of the Leaflet tile pyramid — for embedding arbvis output in docs/PRs without serving a web bundle.

**Approach:**
- `src/lib.rs`: add `#[arg(long = "png", value_name = "FILE")]` to `Args`; conflict with `three_d`, `space`, `regen_html`, and `show_xet_xorbs` (same set the pyramid path handles). In the run routing, when `png` is set and not `diff_mode`, call the new renderer instead of `run_tiles`/the 3D path. Bail with a clear message when combined with `--3d`, `--diff`, or `--show-xet-xorbs` (follow-ups can lift those).
- A new module created by the implementer (it does not exist yet in the tree; suggested location: sibling to `tiled/leaf.rs`, e.g. `tiled/single.rs`):
  - Derive the same geometry as `build_tile_plan` (`src/tiled/mod.rs` ~lines 501–510): `s` from `total`, `kh = s/2`, `kw = s.div_ceil(2)`, image is `(1<<kw) × (1<<kh)`, `square_pixels = height²`, `num_squares = 1 << (kw - kh)`.
  - For each Hilbert index `i` in `0..total`, map to (x, y) with `geometry::hilbert_to_xy_u64(i, kh)` (note: per-square offsets mirror `render_leaf_tile_from_buf`'s `sq_off`/`xy2h_u64` scheme — square `sq` at y-offset 0..., matching the tile layout; keep the mapping consistent with the pyramid so the PNG matches the viewer), color with `color::build_pixel_lut()`, and write via the `png` crate already in Cargo.toml (indexed PNG, palette = the 5 distinct LUT colors — `encode_indexed_png` in `src/tiled/leaf.rs` shows the pattern).
  - Read input bytes through the existing `Data` handles / mmap path (`src/data.rs`, `load_source_data`) so a multi-GB file doesn't get slurped into RAM; iterate byte indices sequentially so mmap pages stay warm.
- Reuse `--out` semantics: with `--png`, `--out` (if given) is the parent dir; default writes `./<input-stem>.png` next to the input. Write to a temp file and rename on success.

**Files touched:** `src/lib.rs`, `src/tiled/mod.rs` (only to export the geometry constants or a small helper), one new source file in `src/tiled/` plus `mod` registration in `src/tiled/mod.rs`, README `## Other useful flags` + `## Quick start`.

**Acceptance criteria:**
- `cargo test` passes, including a new test in the new module (or `#[cfg(test)]` there): render a known small buffer (e.g. 0x00 fill + a 0x41 run), decode the PNG with the `png` crate, assert image dims equal `(1<<kw, 1<<kh)` and that sampled pixels match `build_pixel_lut()` for the expected Hilbert indices (use `geometry::hilbert_to_xy_u64` to compute expected positions).
- A test asserts the CLI rejects `--png` with `--3d` / `--diff` (clap `conflicts_with` or explicit bail covered by a test in `src/lib.rs` tests).
- `arbvis <file> --png out.png` produces a valid PNG for a small real file (manual check documented in the run summary, not a committed fixture).

**Sizing:** one file of new code (~150–200 lines) plus flag wiring and tests — one run. Diff-mode PNG is a possible follow-up plan; do not attempt it here.

_None yet._
