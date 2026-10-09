# Plans

Planned features, written by the plan loop and implemented by the feature loop.
Each plan: goal, approach, files touched, acceptance criteria. Move finished plans to Done.

## Planned

### 3D file-boundary overlays (wireframe boxes per source, toggleable) 

**Goal:** close the first README-deferred 3D item (README "Known limitations"/roadmap bullet "3D file-boundary overlays"): in `--3d` byte mode with multiple sources, show where each input file sits in the Hilbert cube as toggleable wireframe boxes, so section boundaries produce a recognizable 3D signature the same way `geometry::file_rects` outlines files in 2D.

Confirmed absent: no `boundaries`/overlay code exists in `src/volume/` or `template.rs` (grep for `boundar`/`overlay` finds only brick/slab-boundary comments); the byte floor's `manifest()` returns empty and `VolumeLabel` is pick-only.

**Approach:**
- `src/geometry.rs`: add `pub fn decompose_hilbert_range(start: u64, len: u64, order: u32) -> Vec<([u32; 3], u32)>` — decompose the Hilbert-index range `[start, start+len)` into maximal aligned octree nodes (greedy: while remaining, take the largest k where the running start is 8^k-aligned and remaining ≥ 8^k; the node's voxel origin comes from the existing `hilbert3d_node_origin` and its side is the matching power of two; map the node's Hilbert index through `hilbert_d2xyz` for the origin when `hilbert3d_node_origin`'s indexing doesn't line up — verify which fits with a round-trip test). Return `(origin, side)` pairs; a single-file span of the whole cube yields one box.
- `src/volume/mod.rs` (`render_volume`, where `manifest = shape.manifest()` is built and `meta.json` is assembled): for byte shapes (`shape.is_byte_volume()`), compute per-source boxes from `cumulative_offsets` + source lengths via the new helper and add a `boundaries` field to the meta JSON: `[{ "name": <source name>, "boxes": [{"x0","y0","z0","x1","y1","z1"}] }]` (reuse/extend `VoxelBox`'s `Serialize`, which already derives `Serialize`). Structured shapes keep emitting no `boundaries` (their `VolumeEntity` bboxes already serve this). Guard: skip the field (or emit an empty array) when there is a single source covering the whole cube — one giant wireframe box is noise.
- `src/volume/html/template.rs`: when `meta.boundaries` is present and non-empty, build `THREE.LineSegments` wireframe boxes (12-edge segments per box, `EdgesGeometry` or a hand-rolled edge index), one color per source via the existing hue scheme — mirror how the 2D viewer colors file outlines if that's reachable from the template, else use `geometry::name_hue`-style HSL computed client-side from the name. Add a checkbox toggle in the control panel (default off, like the other optional overlays) and include the box in `load()`'s meta handling next to `manifestG`. Keep all additions in the meta-handling + control-panel regions; do not touch the ray-march shader.

**Files touched:** `src/geometry.rs`, `src/volume/mod.rs`, `src/volume/html/template.rs`. No new dependency (std + three.js, already the viewer's renderer).

**Acceptance criteria:**
- `cargo test` passes, including new tests: (a) `decompose_hilbert_range` round-trip — for every box and every Hilbert index in its node range, `hilbert_d2xyz` lands inside the box, boxes are disjoint and their total index length equals `len` (full-cube, straddling, and multi-source ranges, e.g. boundaries at 7_777 like `src/tiled/leaf.rs`'s boundary tests); (b) a `render_volume` (or meta-assembly-level) test asserting two-source byte runs write a `boundaries` array with one entry per source and a single-source run writes none/empty.
- Template string tests in `src/volume/html/mod.rs`: the template references `meta.boundaries`, guards on empty, and wires the toggle checkbox (assert on the checkbox id + the guard, following the existing `template_tests` style).
- Manual check in the run summary: `arbvis a.bin b.bin --3d out-dir` writes `meta.json` containing `boundaries`, and the viewer opens with the toggle rendering two wireframe boxes (or note it if a browser check is impractical; the template assertions then carry the behavior).

**Sizing:** ~200–300 lines across three files including tests — one run. The sibling deferred item (interactive transfer-function editor with density histogram) is NOT part of this plan; it should get its own plan later.


## Done

### Xet-mode single-image PNG export (`--png FILE` with `--show-xet-xorbs`) — Done 2026-10-09

**Goal:** let `arbvis --show-xet-xorbs <hf-file> --png out.png` write the xorb-colored render as one PNG — the last remaining `--png` conflict (diff-mode PNG already shipped; `--3d`/`--space`/`--regen-html` conflicts stay, since PNG is a 2D single-image export).

**Approach:**
- `src/cli.rs`: remove `"show_xet_xorbs"` from the `--png` `conflicts_with_all` list (~line 162) and from the `png_conflicts_with_3d_space_regen_and_xorbs` test's flag list (`mod png_flag_tests` ~line 608) — the test then asserts `--png` still conflicts with `--3d`, `--space`, `--regen-html` and composes with `--show-xet-xorbs`. Update the `--png` help text (the `///` doc on the field) to mention xorb mode.
- `src/tiled/mod.rs`: factor the XorbMap construction out of `build_tile_plan` (the `let xorb_map = if show_xet_xorbs { XorbMap::build(...) }` block around lines 386–394) into `pub(super) fn xet_xorb_ranges(sources: &[Source], cumulative_offsets: &[u64]) -> XorbMap` — same shape as the existing `diff_leaf_mode` helper. `build_tile_plan` calls it; the new renderer calls it too. Note the Tableau palette for the renderer is already module-level (`TABLEAU_20` → `tableau` at ~line 401); expose or reuse it rather than duplicating.
- `src/tiled/single.rs`: add `pub async fn render_single_xet_png(sources: &[Source], total: u64, out: &Path) -> anyhow::Result<()>`, mirroring `render_single_diff_png` (same `single_geometry` + `open_sources` + `load_tile_bytes` tile loop + `blit_tile` + `encode_rgb_png` + tmp-then-rename) but rendering each tile with `leaf::render_leaf_tile_xet_from_buf(tx, ty, geom.kh, height_tiles, geom.square_pixels, total, &tile_buf, &pixel_lut, &xorb_ranges, &tableau, TileFormat::Png)` (`src/tiled/leaf.rs` ~line 554). `pixel_lut` is `color::build_pixel_lut()` — check the exact name via the `render_single_png`/diff-path imports. Output is truecolor RGB (xet scales Tableau colors per byte, so no 256-entry palette) → `encode_rgb_png`, not the indexed encoder.
- `src/pipeline.rs`: in the `--png` routing block (~lines 193–208), add a branch ahead of the plain branch: when `hints.show_xet_xorbs`, call `render_single_xet_png`. If the built `XorbMap` is empty (no source had xet terms — e.g. a local file), log the same kind of warning the `--3d` path already emits (~line 82) and fall back to `render_single_png`. Sources' `xet_terms` are already populated before this point: `chosen.prepare(&ctx)` (which calls `data::populate_xet_terms`, `src/providers.rs` ~line 172) runs before the PNG routing.

**Files touched:** `src/cli.rs`, `src/pipeline.rs`, `src/tiled/mod.rs` (helper extraction only), `src/tiled/single.rs`. No new module, no new dependency.

**Acceptance criteria:**
- `cargo test` passes, including a new test in `src/tiled/single.rs`'s `#[cfg(test)]` module: render an xet PNG for fabricated sources with synthetic `xet_terms` (the `XorbMap::build` tests in `src/xet/mod.rs` ~line 929 show the term-fabrication pattern), decode it with the `png` crate, and assert sampled pixels match the Tableau-scaled colors computed directly from `render_leaf_tile_xet_from_buf`'s per-tile output blitted at `(tx*TILE, ty*TILE)`.
- The extraction of `xet_xorb_ranges` keeps `build_tile_plan`'s existing tests green unmodified (same guard style as the diff refactor).
- A `src/cli.rs` test asserts `--png` + `--show-xet-xorbs` parses without conflict and `--png` still conflicts with `--3d`, `--space`, `--regen-html`.
- Manual check documented in the run summary: `arbvis <local file> --show-xet-xorbs --png out.png` warns and writes the plain indexed PNG; a remote hf source with xet terms writes a valid truecolor PNG (or, if remote fetch is impractical in the run, note that and rely on the fabricated-terms test).

**Sizing:** ~150–200 lines across four existing files plus tests — one run. With this landed, `--png` conflicts shrink to `--3d`, `--space`, `--regen-html`.

**Implemented 2026-10-09 by feature.** `render_single_xet_png` returns `Ok(bool)` (false when the built `XorbMap` is empty) so the pipeline can warn and fall back to plain mode; the Tableau palette was factored into `pub(super) fn tableau_palette()` alongside `xet_xorb_ranges`. Manual check: `arbvis <local file> --show-xet-xorbs --png out.png` warned "no xet/xorb ranges found" and wrote the plain indexed PNG; the remote-hf truecolor path is covered by the fabricated-terms test (`renders_xet_xorbs_as_truecolor_png`).


### JSON parser error-path and escape-decoding test coverage — Done 2026-10-09

**Implemented 2026-10-09 by coverage.** `src/json_diff/parse.rs` had only two error tests
(`error_position`, `error_trailing_data`); the string/number/literal error branches and the
`\uXXXX` surrogate handling were untested. Added 8 tests: surrogate-pair key decoding
(U+1F600 via `\uD83D\uDE00`), basic escape decoding (`\n\t\b\f\r\/\\\"`), string error paths
(unterminated, EOF in escape, invalid escape, unescaped control byte, invalid hex digit,
EOF in `\uXXXX`, unexpected low surrogate, missing low surrogate, invalid low surrogate,
invalid UTF-8 lead/continuation byte, malformed multi-byte → U+FFFD), number error paths
(missing integer part, empty fraction, empty exponent, valid `-0.5e-3`/`1E+4`), literal error
paths (bool/null/empty input), and structural error paths (missing key quote, trailing commas,
missing value). Also fixed a pre-existing `clippy -D warnings` failure: empty line after doc
comment in `src/volume/brick_stream.rs` (`track_occupied_brick` doc).

**Verified 2026-10-09:** `cargo test --lib` green (0 failed); `cargo clippy --lib --tests --
-D warnings` clean.
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
