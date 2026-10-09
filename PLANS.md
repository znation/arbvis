# Plans

Planned features, written by the plan loop and implemented by the feature loop.
Each plan: goal, approach, files touched, acceptance criteria. Move finished plans to Done.

## Planned

_None yet._

## Done

### Diff-mode single-image PNG export (`--png FILE` with `--diff`) — Done 2026-10-09

**Implemented 2026-10-09 by feature.** As planned. One clarification: the factored helper is
`pub(super) fn diff_leaf_mode(sources: &[Source]) -> LeafMode` (no `total` parameter — it reads
`byte_size` off each source). The diff PNG is rendered per TILE×TILE tile through the pyramid's
`load_tile_bytes` + `render_leaf_tile_diff` (TileFormat::Png), blitted at `(tx*TILE, ty*TILE)`,
and encoded truecolor RGB via a new `encode_rgb_png`. Manual check: `arbvis --diff orig.bin
mod.bin --png out.png` on 5000-byte random pairs wrote a valid 512×512 8-bit RGB PNG, and plain
`--png` still writes the indexed (colormap) PNG.

**Verified 2026-10-09:** the single-image PNG path shipped (`src/tiled/single.rs`, routed in `src/pipeline.rs` ~line 184) deliberately excludes diff: `src/cli.rs` `--png` has `conflicts_with_all = ["three_d", "diff", "space", "regen_html", "show_xet_xorbs"]` (~line 159), and no diff-rendering code exists in `single.rs`. `render_single_png` uses only the plain byte LUT.

**Goal:** let `arbvis --diff ORIGINAL MODIFIED --png out.png` write the diff visualization as one PNG, so diff output can be embedded in docs/PRs without serving a web bundle (same motivation as the original `--png` plan).

**Approach:**
- `src/cli.rs`: remove `"diff"` from the `--png` `conflicts_with_all` list; keep the other conflicts. Update the `png_conflicts_with_3d_diff_space_regen_and_xorbs` test (rename/adjust to drop the `--diff` case and add an assertion that `--png` + `--diff` is now accepted).
- `src/tiled/mod.rs`: factor the diff-mode `LeafMode::Diff { pixel_lut, plain_lut, fills, tints }` construction out of `build_tile_plan` (the `build_diff_signed_lut()` branch ~line 303 and the crosshatch fills/tints collection ~lines 409–451) into a `pub(super)` helper, e.g. `fn diff_leaf_mode(sources: &[Source], total: u64) -> LeafMode`. `build_tile_plan` calls it in place of the inline code; the new renderer calls it too.
- `src/tiled/single.rs`: add `pub async fn render_single_diff_png(sources: &[Source], total: u64, fills/tints (or the LeafMode), out: &Path) -> anyhow::Result<()>`:
  - Geometry via the existing `single_geometry(total)` and the tile scatter helpers already in this file — the tile grid is identical in diff mode.
  - For each tile `(tx, ty)`: load the signed-delta tile buffer with `tiled::leaf::load_tile_bytes` (async, handles local mmap and remote `Data` variants — same loading path the diff pyramid uses via `HilbertBytesLoader`), then render with `tiled::leaf::render_leaf_tile_diff(tx, ty, kh, height_tiles, square_pixels, total, &tile_buf, &pixel_lut, &plain_lut, &fills, &tints, ...)`. That returns a `TileResult` carrying a `TILE×TILE` RGB image; blit it into a full `ImageBuffer<Rgb<u8>, Vec<u8>>` at raster `(tx*TILE, ty*TILE)`.
  - Encode as a truecolor RGB PNG, not indexed: diff output mixes signed-delta LUT colors, one-sided-source tints, and crosshatch fills, so the palette is not 256 entries. Add a small `encode_rgb_png(width, height, &img)` alongside the existing `encode_indexed_single_png` (the `png` crate is already a dependency).
  - Reuse `png_output_path` and the write-to-temp-then-rename pattern of `render_single_png`.
- `src/pipeline.rs`: in the `--png` routing block (~line 184), when `hints.diff_mode` is true call `render_single_diff_png` instead of `render_single_png`; the sources/total from `chosen.prepare` already carry the diff pair. The existing `total == 0` bail stays.

**Files touched:** `src/cli.rs`, `src/pipeline.rs`, `src/tiled/mod.rs` (helper extraction only), `src/tiled/single.rs`. No new module, no new dependency.

**Acceptance criteria:**
- `cargo test` passes, including new tests in `src/tiled/single.rs`'s `#[cfg(test)]` module: (1) render a diff PNG for a small known original/modified pair with a fabricated fill/tint list, decode it with the `png` crate, and assert sampled pixels match a reference computed directly from `render_leaf_tile_diff`'s per-tile output placed at `(tx*TILE, ty*TILE)`; (2) the factored-out `diff_leaf_mode` helper produces the same `LeafMode::Diff` fills/tints `build_tile_plan` produced before the extraction (guard the refactor with the existing `build_tile_plan` tests, which must stay green unmodified).
- A `src/cli.rs` test asserts `--png` + `--diff` parses without conflict and `--png` still conflicts with `--3d`, `--space`, `--regen-html`, `--show-xet-xorbs`.
- `arbvis --diff orig mod --png out.png` produces a valid PNG for a small real pair (manual check documented in the run summary, not a committed fixture).

**Sizing:** ~150–200 lines across four existing files plus tests — one run. `--png` for `--show-xet-xorbs` remains a separate possible follow-up; do not attempt it here.

_Plan written 2026-10-09._

### Single-image PNG export (`--png FILE`) — Done 2026-10-09

**Implemented 2026-10-09 by feature.** As planned, with two clarifications: in PNG mode `OutputDest::from_args` is skipped entirely (so `--png` needs no `--out`; a given `--out DIR` is the PNG's parent, created if missing, and `hf://` outs are rejected); and the indexed PNG uses the full 256-entry `build_pixel_lut()` as its palette, so pixel index = byte value.

**Found by plan loop 2026-10-09.** Verified 2026-10-09: no single-image export exists — `src/lib.rs` `Args` has only `--tile-format` (tiles), and `grep -rni 'single.image\|snapshot' src README.md` shows no PNG-export path.

**Goal:** add a `--png FILE` flag that writes the entire 2D render as one PNG file (one pixel per byte, same byte-color scheme) instead of the Leaflet tile pyramid — for embedding arbvis output in docs/PRs without serving a web bundle.

**Approach:**
- `src/lib.rs`: add `#[arg(long = "png", value_name = "FILE")]` to `Args`; conflict with `three_d`, `space`, `regen_html`, and `show_xet_xorbs` (same set the pyramid path handles). In the run routing, when `png` is set and not `diff_mode`, call the new renderer instead of `run_tiles`/the 3D path. Bail with a clear message when combined with `--3d`, `--diff`, or `--show-xet-xorbs` (follow-ups can lift those).
- A new module created by the implementer (it does not exist yet in the tree; suggested location: sibling to `tiled/leaf.rs`, e.g. `tiled/single.rs`):
  - Derive the same geometry as `build_tile_plan` (`src/tiled/mod.rs` ~lines 501–510): `s` from `total`, `kh = s/2`, `kw = s.div_ceil(2)`, image is `(1<<kw) × (1<<kh)`, `square_pixels = height²`, `num_squares = 1 << (kw - kh)`.
  - For each Hilbert index `i` in `0..total`, map to (x, y) with `geometry::hilbert_to_xy_u64(i, kh)` (note: per-square offsets mirror `render_leaf_tile_from_buf`'s `sq_off`/`xy2h_u64` scheme — square `sq` at y-offset 0..., matching the tile layout; keep the mapping consistent with the pyramid so the PNG matches the viewer), color with `color::build_pixel_lut()`, and write via the `png` crate already in Cargo.toml (indexed PNG, palette = the 5 distinct LUT colors — `encode_indexed_png` in `src/tiled/leaf.rs` shows the pattern).
  - Read input bytes through the existing `Data` handles / mmap path (`src/data/mod.rs`, `load_source_data`) so a multi-GB file doesn't get slurped into RAM; iterate byte indices sequentially so mmap pages stay warm.
- Reuse `--out` semantics: with `--png`, `--out` (if given) is the parent dir; default writes `./<input-stem>.png` next to the input. Write to a temp file and rename on success.

**Files touched:** `src/lib.rs`, `src/tiled/mod.rs` (only to export the geometry constants or a small helper), one new source file in `src/tiled/` plus `mod` registration in `src/tiled/mod.rs`, README `## Other useful flags` + `## Quick start`.

**Acceptance criteria:**
- `cargo test` passes, including a new test in the new module (or `#[cfg(test)]` there): render a known small buffer (e.g. 0x00 fill + a 0x41 run), decode the PNG with the `png` crate, assert image dims equal `(1<<kw, 1<<kh)` and that sampled pixels match `build_pixel_lut()` for the expected Hilbert indices (use `geometry::hilbert_to_xy_u64` to compute expected positions).
- A test asserts the CLI rejects `--png` with `--3d` / `--diff` (clap `conflicts_with` or explicit bail covered by a test in `src/lib.rs` tests).
- `arbvis <file> --png out.png` produces a valid PNG for a small real file (manual check documented in the run summary, not a committed fixture).

**Sizing:** one file of new code (~150–200 lines) plus flag wiring and tests — one run. Diff-mode PNG is a possible follow-up plan; do not attempt it here.

_None yet._
