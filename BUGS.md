# Bugs

Known bugs, recorded by any loop and fixed by the bugfix loop.
Each bug: symptom, how to reproduce, suspected cause if known. Move fixed bugs to Fixed, with the
required `**Validation gap:** <tag> — <one sentence>` line recording what made the bug hard to
confirm (tag one of: none, no-repro, no-fake, real-run-needed, no-observability, slow-check,
unclear-invariant).

## Open

### `src/volume/html.rs` (1889 lines) exceeds the one-sitting readability budget

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: tiled/mod.rs, volume/mod.rs, volume/brick.rs, xet.rs, data/mod.rs).
Caveat recorded on inspection: the file is ~15 lines of Rust plus a ~1860-line embedded
Three.js template string, so a Rust-module split has poor ROI — treat as opportunistic work if a
feature loop already touches the viewer, e.g. extracting the JS shader/uniform blocks behind named
const segments. Do not refactor for refactoring's sake.

**Reproduce:** `wc -l src/volume/html.rs`.

### `src/tiled/mod.rs` (1575 lines) exceeds the one-sitting readability budget

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: volume/html.rs, volume/mod.rs, volume/brick.rs, xet.rs, data/mod.rs).
Symptom: tile pipeline orchestration, progress plumbing, and Leaflet-regeneration glue share one file;
the tile pipeline's geometry constants are spread across ~490–591. Suggested direction: split
cohesive units opportunistically (e.g. `regen_html`/`regen_html_multi` regeneration path from the
render pipeline), one split per tick, with tests moved alongside. Do not refactor for refactoring's sake.

**Split 3 landed 2026-10-09 by feature:** the scene-grouping path (`SceneGroup`,
`sanitize_scene_key`, `partition_scenes`, and the four scene tests) moved to new
`src/tiled/scenes.rs` (191 lines); `tiled/mod.rs` now imports `partition_scenes`/`SceneGroup`
from `scenes` (`use scenes::{…}`) and `tiled/streaming.rs` imports them via
`super::scenes::{…}`. `sanitize_scene_key` became module-private in `scenes.rs` (its only
caller is `partition_scenes`). No callers outside `crate::tiled` use these items.
`tiled/mod.rs` is now 1255 lines — still over budget; next split candidates: phases out of
`drive_pipeline`, or `render_scene_to_disk` out of `run_tiles`.

**Split 2 landed 2026-10-09 by feature:** the regeneration path (`regen_html`, `regen_html_multi`,
`file_entity_from_json`, `sniff_ext_in`, `sniff_ext_for_zoom`) moved to new `src/tiled/regen.rs`
(234 lines); `tiled/mod.rs` re-exports `regen_html`, so external callers (`pipeline.rs`) and docs
are unchanged. No tests lived in the moved block (the `regen_html` error-path test stays in
`mod.rs` scene_tests). `tiled/mod.rs` is now 1424 lines — still over budget; next split candidates:
`SceneGroup`/`sanitize_scene_key`/`partition_scenes` (~1258–1339 in the pre-split file) into a
`scenes` module, or phases out of `drive_pipeline`.

**Reproduce:** `wc -l src/tiled/mod.rs`.

### `src/volume/mod.rs` (1215 lines) exceeds the one-sitting readability budget

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: volume/html.rs, tiled/mod.rs, volume/brick.rs, xet.rs, data/mod.rs).
Suggested direction: split cohesive units opportunistically, one split per tick, with tests moved
alongside. Do not refactor for refactoring's sake.

**Reproduce:** `wc -l src/volume/mod.rs`.

### `src/volume/brick.rs` (1172 lines) exceeds the one-sitting readability budget

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: volume/html.rs, tiled/mod.rs, volume/mod.rs, xet.rs, data/mod.rs).
Suggested direction: split cohesive units opportunistically, one split per tick, with tests moved
alongside. Do not refactor for refactoring's sake.

**Reproduce:** `wc -l src/volume/brick.rs`.

### `src/data/mod.rs` (853 lines) exceeds the one-sitting readability budget

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: volume/html.rs, tiled/mod.rs, volume/mod.rs, volume/brick.rs,
xet.rs). Already reduced by the diff-subsystem extraction (see Fixed 2026-10-09); remainder is the
source/IO half (Data, SourceKind, Source, prepare_sources). Suggested direction: opportunistic
splits only. Do not refactor for refactoring's sake.

**Reproduce:** `wc -l src/data/mod.rs`.

Note: the original combined entry also listed `lib.rs` at 1131 lines, which was stale —
`lib.rs` is 134 lines and in budget.

## Fixed

### `src/xet/mod.rs` (1121 lines before the fetch split) exceeds the one-sitting readability budget — fixed 2026-10-09 by bugfix

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: volume/html.rs, tiled/mod.rs, volume/mod.rs, volume/brick.rs,
data/mod.rs). Split 1 landed 2026-10-09 by bugfix: the protocol unit — wire types
(`XetReadTokenResponse`, `WireRange`, `ReconstructionTerm`, `WireXorbRangeDescriptor`,
`WireXorbMultiRangeFetch`, `ReconstructionResponse`, `CasToken`) plus the CAS-token cache and fetch
helpers (`authed_get_json`, `fetch_cas_token`, `fetch_reconstruction_response`,
`fetch_reconstruction_terms`, `invalidate_cas_token_cache`) — moved verbatim to new `src/xet/fetch.rs`
(211 lines), with the moved items marked `pub(super)` so the `crate::xet` public surface
(`XetTerm`, `reconstruction_for`, `XetReader`, `XorbMap`, `TABLEAU_20`, `CasStats`, `http_client`)
is unchanged; `lib.rs` untouched. `src/xet/mod.rs` is now 946 lines — still over budget; next split
candidate: the `XetReader` impl + its cache/refresh support into a sibling reader module.

**Reproduce (was):** `wc -l src/xet/mod.rs` (1121 lines before the split; the module was single-file then). Now: mod.rs 946, fetch.rs 211 lines.

**Validation gap:** unclear-invariant — the split is a pure code move, so a passing suite could not
distinguish a faithful move from one that silently changed visibility or the public API; had to
reconstruct the `crate::xet` surface contract first (grep over external callers, cargo build errors
driving the `pub(super)` markings).

### `src/data/mod.rs` (853 lines) exceeds the one-sitting readability budget — fixed 2026-10-09 by bugfix

**Found by steward 2026-10-09; split into per-file entries by bugfix 2026-10-09** (from the combined
"several modules" entry; siblings: volume/html.rs, tiled/mod.rs, volume/mod.rs, volume/brick.rs,
xet.rs). Fixed in two ticks: first the diff subsystem was extracted into `src/data/diff.rs` (see
the earlier Fixed entry); this tick split the remainder into `src/data/source.rs` (local source
construction: `collect_files_recursive`, `prepare_sources`, `load_source_data`, and the
`prepare_sources_tests` moved alongside) and `src/data/remote.rs` (Hub plumbing:
`prepare_sources_from_specs`, `materialize_http_sources`, `download_specs_to_paths`,
`populate_xet_terms`). `src/data/mod.rs` now holds only the shared type definitions (Data,
SourceKind, Source, Extensions, SceneTag, InputSpec, CustomSource, LazyFetcher) plus re-exports,
so the `crate::data::{...}` surface and `lib.rs` are unchanged. Code was moved verbatim, with one
exception: `diff_bytes_to_color` now lives in `src/data/source.rs` next to its two callers and its
`diff_bytes_to_color_tests` module moved with it (revision 2 restored the function and its tests,
which had been dropped and its body inlined into two source.rs closures).

**Reproduce (was):** `wc -l src/data/mod.rs`. Now: mod.rs 318, source.rs 293, remote.rs 340,
diff.rs 492 lines — each in budget.

**Validation gap:** none — `wc -l` confirms the split, and the existing suite (including the
prepare_sources tests and the diff_bytes_to_color_tests in source.rs) plus `cargo build`/
`cargo fmt --check` passed after the change.

### Extract the diff subsystem of the former single-file data module into `src/data/diff.rs` — fixed 2026-10-09 by bugfix

**Found by steward 2026-10-09** as part of the combined "several modules exceed the one-sitting
readability budget" entry (now decomposed into per-file entries under Open; this tick's slice:
the data module). The diff subsystem — `DiffFill`, the built-in `DiffSourceBuilder`s
(`JsonDiffBuilder`, `PlainBytesDiffBuilder`), `prepare_diff_sources`, and the directory byte-diff
walker (`byte_directory_diff`, `collect_files_recursive`) — was ~370 of the original file's 1203 lines. The remainder is now `src/data/mod.rs` (853 lines). External paths are unchanged: `data/mod.rs` re-exports the moved
items, so `crate::data::DiffFill`, `crate::data::byte_directory_diff`, the lib re-exports, and
downstream (`tiled`, `json_diff`, `registry`, modelweightvis) all resolve as before. No behavior
change intended; new `data::diff::tests` cover `DiffFill` colors, matched/mismatched/one-sided
classification in `byte_directory_diff`, finetune grey fill, the `skip` predicate, and builder
ids/priority ordering.

**Validation gap:** unclear-invariant — the suite verified compile-compatibility of the move but the
diff walker itself had no direct tests, so the classification invariants (padding, one-sided fill
choice, skip semantics, compacted `file_idx`) had to be reconstructed before they could be pinned.
