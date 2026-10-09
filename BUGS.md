# Bugs

Known bugs, recorded by any loop and fixed by the bugfix loop.
Each bug: symptom, how to reproduce, suspected cause if known. Move fixed bugs to Fixed, with the
required `**Validation gap:** <tag> — <one sentence>` line recording what made the bug hard to
confirm (tag one of: none, no-repro, no-fake, real-run-needed, no-observability, slow-check,
unclear-invariant).

## Open

### Several modules far exceed the one-sitting readability budget

**Found by steward 2026-10-09.** `wc -l` over `src/`: `volume/html.rs` 1889, `tiled/mod.rs` 1583, `volume/mod.rs` 1215, `data.rs` 1203, `lib.rs` 1131, `xet.rs` 1121, `volume/brick.rs` 1172 — all well past PRINCIPLES.md's "small enough to read in one sitting." Symptom: new contributors (and the plugin registry downstream, e.g. modelweightvis) must hold whole large files in mind to find the seams; the tile pipeline's geometry constants are spread across `tiled/mod.rs` ~490–591 while `lib.rs` mixes CLI parsing, routing, and orchestration.

Risk, not a runtime bug: no behavior change proposed. Suggested direction if a feature loop is already touching one of these files — split out cohesive units opportunistically (e.g. `volume/html.rs` template/markup generation from its data plumbing, `lib.rs` `Args`/routing separation), one split per tick, with tests moved alongside. Do not refactor for refactoring's sake.

**Reproduce:** `wc -l src/*.rs src/*/*.rs | sort -rn | head`.

## Fixed

_None yet._
