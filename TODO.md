# Outstanding work after the `arbvis` / `modelweightvis` split

The two-crate split landed across branch `zn/focused-mclaren-c3e069` (this
repo) and the standalone `~/hf/modelweightvis` repo. Every hook trait
arbvis exposes has a concrete impl on the modelweightvis side; arbvis
byte-only output is byte-identical to the step-12e baseline; `cargo
test --workspace` is 169 tests green; `cargo clippy --workspace` is
zero warnings.

What's listed below didn't block landing the split but is still real
work. Roughly ordered by impact-per-effort.

## Workspace (`~/hf/arbvis`, branch `zn/focused-mclaren-c3e069`)

### 1. ~~Wire `FormatPlugin::populate_remote`~~ — DONE

**Status.** Wired. `prepare_sources_from_specs` in
[`crates/arbvis/src/data.rs`](crates/arbvis/src/data.rs) is now `async`;
the `InputSpec::Remote` arm constructs a `Data::Http` handle and
iterates `registry.formats` to call `populate_remote` on the first
plugin whose `detects_path` matches. The single caller in `lib.rs`'s
`resolve_input_sources` was updated to `.await` the function. Failures
are logged and treated as "no plugin populated extensions" (same
fallback the `populate_local` block uses); pickle remote bails out by
design.

Verification: `cargo clippy --workspace --all-targets -- -D warnings`
zero warnings, `cargo test --workspace` 87 passed. The local-default
path (`prepare_sources`, the non-`--stream` non-`--show-xet-xorbs` case)
was not touched, so local smoke baselines remain byte-identical.

Remote end-to-end pixel verification is still pending and tracked as
item 4 below — that's the only thing left to confirm this change has
the intended effect on `--stream` arch runs.

### 2. ~~Wire `SourceMeta` sidecar enrichment~~ — DONE

**Status.** Wired via the first of the two options from the original
plan: a new arbvis hook `PrepareSourcesExtension` in
[`crates/arbvis/src/registry.rs`](crates/arbvis/src/registry.rs) that
runs once per render at the top of `dispatch_render`, after every
`Source` has been built. The new
[`SourceMetaSidecarHook`](crates/modelweightvis/src/hooks.rs) (registered
by [`register_all`](crates/modelweightvis/src/lib.rs)) implements the
trait by calling `load_meta_for_sources(sources).await` and inserting
each `SourceMeta` into the corresponding `Source.extensions` slot.

`ArchLayoutPlugin::build` in
[`crates/modelweightvis/src/layout/mod.rs`](crates/modelweightvis/src/layout/mod.rs)
now pulls a parallel `Vec<SourceMeta>` back out of every source's
extensions and threads it into `ArchLayout::try_build` (replacing the
`&[]` placeholder). The `#[allow(dead_code)]` markers on
`try_load_source_meta` / `load_meta_for_sources` / `fetch_hf_sidecar`
in [`data.rs`](crates/modelweightvis/src/data.rs) are gone.

Behaviour change: `config.json`'s `num_hidden_layers` now extends the
arch canvas's layer stack to the declared count (so partial-shard loads
produce a stable layout), `architectures` populates
`ArchLayout::architecture` for downstream display, and
`model.safetensors.index.json` (when present) seeds canonical sub-path
slots for tensors that live in shards we didn't load.

Verification: `cargo build --workspace` clean, `cargo clippy --workspace
--all-targets -- -D warnings` zero warnings, `cargo test --workspace`
169 tests (82 arbvis + 87 modelweightvis) passed. Local smoke run of
`modelweightvis $SMOLLM2/model.safetensors --output …` produces a valid
4096×4096 PNG; debug log shows `config.json` loaded from the sidecar
path, confirming the hook fires end-to-end.

Note: the standalone `~/hf/modelweightvis` repo (TODO item 6) is still
on the pre-hook code and would need a parallel mirror commit once the
path-dep flip happens.

### 3. ~~Move model-side CLI flags off `arbvis::Args`~~ — DONE

**Status.** Wired. The four tensor-aware flags (`--moe-diff`,
`--finetune` / `--no-finetune`, `--diff-metric`, `--layout`) and their
clap-mirror enums (`DiffMetricArg`, `LayoutArg`) have moved from
`arbvis::Args` to a new
[`modelweightvis::Args`](crates/modelweightvis/src/args.rs) that
clap-flattens the byte-only `arbvis::Args` and adds those four. The
new `arbvis::ModelOpts` (a plain data struct, no clap derive) carries
the four runtime knobs through to `arbvis::run`; arbvis's binary
passes `ModelOpts::default()` and modelweightvis's binary calls
`Args::split()` to peel out the inner `(arbvis::Args, ModelOpts)`
pair.

`arbvis::run` signature now takes `(args, opts, registry)` (was
`(args, registry)`). The body reads `opts.moe_diff`, `opts.finetune`,
`opts.no_finetune`, `opts.diff_metric`, and `opts.layout_mode` in
place of the deleted `args.*` field accesses.

Help-text proof:
```
$ arbvis --help | grep '^\s*--'
   --diff, --space, --regen-html, --title, --show-xet-xorbs,
   --tile-format, --stream
$ modelweightvis --help | grep '^\s*--'
   --diff, --space, --regen-html, --title, --show-xet-xorbs,
   --tile-format, --stream, --moe-diff, --finetune, --no-finetune,
   --diff-metric, --layout
```

Cross-flatten clap constraints work: `--moe-diff` (defined on
`modelweightvis::Args`) successfully `conflicts_with` `--diff` /
`--files` / `--file_list` / `--show_xet_xorbs` (all flattened from
`arbvis::Args`); `--finetune` / `--no-finetune` correctly `requires =
"diff"` and conflict with each other. The pre-existing clap quirk
where a positional `[FILES]` value bypasses `requires = "diff"` is
preserved verbatim (same behaviour as before the move).

Verification: `cargo fmt --all -- --check` clean, `cargo clippy
--workspace --all-targets -- -D warnings` zero warnings, `cargo test
--workspace` 169 tests passed. Smoke runs of `arbvis /bin/ls --output`
(byte-only path) and `modelweightvis $SMOLLM2/model.safetensors
--output` (tensor-aware path with `SourceMetaSidecarHook` firing)
both produce valid PNGs.

Note: same caveat as item 2 — the standalone `~/hf/modelweightvis`
repo (item 6) is still on the pre-split CLI surface (it uses
`arbvis::Args::parse()` directly) and would need a parallel mirror
once the path-dep flip lands.

### 4. ~~End-to-end smoke verification for remote scenarios~~ — DONE

**Status.** All four remote-scenario shapes exercised end-to-end
post-relocation. No HF_TOKEN was available in this environment so
the gated Llama-3.2 / Mixtral scenarios listed below were substituted
with open repos that already had partial-to-full HF cache state
(scenarios produce equivalent path coverage; no Llama-specific code
exists in arbvis or modelweightvis).

No `/tmp/arbvis-step12e` pixel baseline existed, so the bar was
"valid PNG / tile pyramid + no panics + hooks fire as expected
(debug log)". Pixel statistics confirm each output is non-trivial.

Scenarios run (substitutions in brackets):

| # | Command | Time | Output | Pixel stats |
|---|---------|------|--------|-------------|
| 1 | `modelweightvis hf://HuggingFaceTB/SmolLM2-135M --tiles /tmp/mw-sl`  [Llama-3.2-1B] | 42.6 s | 2048 leaf PNG + 682 AVIF pyramid + index.html | (single PNG variant: 4096², mean 34.5 σ 55.3) |
| 2 | `modelweightvis --diff hf://HuggingFaceTB/SmolLM2-135M hf://HuggingFaceTB/SmolLM2-135M-Instruct --tiles /tmp/mwd-sl`  [Llama-3.2-1B vs -Instruct] | 65.5 s | 4568 leaf PNG + 1365 AVIF pyramid + zoom-11 detail tiles | non-trivial; finetune auto-detect fired |
| 3 | `modelweightvis --moe-diff hf://Qwen/Qwen1.5-MoE-A2.7B --output /tmp/mwmoe-qwen.png`  [Mixtral-8x7B] | 12 m 24 s | 4096² PNG, 15 MiB; 24 layers × 60² experts × 3 projections = 131 760 cells, 380 GB synthetic diff bytes | mean 42.5 σ 70.9 |
| 4 | `modelweightvis --diff /tmp/dirdiff-a /tmp/dirdiff-b --no-finetune --output /tmp/dirdiff.png` (synthetic local dirs: pristine vs 64-byte-mutated `model.safetensors` + size-mismatched `config.json` + appended `README.md` + unchanged `tokenizer.json`) | 0.8 s | 4096² PNG, 483 KiB | mean 0.27 σ 3.9 (tiny diff highlight as designed) |

**Hooks confirmed firing end-to-end:**
- `SourceMetaSidecarHook` — scenario 1 debug log: `loaded
  config.json from /Users/.../models--HuggingFaceTB--SmolLM2-135M/.../config.json`
  (sidecar populated via the new `PrepareSourcesExtension` slot).
- `populate_remote` (FormatPlugin) — scenarios 1/2/3 all
  `hf://`-rooted; the `prepare_sources_from_specs` populate loop
  fired (extensions present on every source going into `ArchLayoutPlugin`).
- `TensorRepoDiffPrep` — scenario 2 paired up safetensors halves
  across the two repos; finetune auto-detect (no `--finetune` /
  `--no-finetune` passed) resolved as finetune, producing the
  expected green-crosshatch warning for 15 instruct-only files
  (`onnx/*.onnx`, `runs/...tfevents.*`, `trainer_state.json`,
  `training_args.bin`, etc.) and 5 size-mismatch byte-diff
  warnings (`README.md`, `config.json`, `generation_config.json`,
  `special_tokens_map.json`, `tokenizer_config.json`).
- `TensorMoeDiffPrep` — scenario 3 emitted the canonical
  `moe-diff: 24 layer(s), 3 weight slot(s) per layer (gate_proj
  up_proj down_proj), emitted 131760 cell(s)` log line; render
  finished cleanly.
- `TensorDirectoryDiffPrep` — scenario 4 paired 272 tensors in
  pass-0 exact match; non-safetensors files (`README.md`,
  `config.json`) flagged at 3 entries with the canonical "arch
  layout: N non-safetensors diff source(s) will not appear on the
  arch canvas" log.
- `materialize_remote_arcs` (restored in `4c0e7be`) — scenarios
  1/2/3 all reach it on the Source-prep path; no panics, no
  spurious refetches.

**Caveats / observations:**
- Scenario 3 + 4 use `--output` (single-image) for runtime bounding
  → trigger the documented `architectural single-image layout
  requires local non-diff non-xet inputs; falling back to hilbert`
  fallback. That fallback is correct behaviour (diff/MoE-diff
  sources are synthetic and not arch-renderable as a single image);
  `--tiles` produces the proper arch / MoE-diff canvas.
- Substituting open repos for the TODO's gated Llama / Mixtral
  examples doesn't change path coverage — `modelweightvis` has no
  per-model branches; the size, shard count, and MoE topology are
  what stress the hooks, and the substituted repos cover those
  axes (SmolLM2: single-file safetensors; Qwen1.5-MoE: 8-shard
  safetensors with HF-style routed experts). Mixtral-style
  per-expert fused-tensor GGUF coverage is documented as
  unsupported in the help text for `--moe-diff`.
- The standalone `~/hf/modelweightvis` repo is still on
  pre-#1/#2/#3 source (item 6) and would benefit from the same
  smoke run once the path-dep flip lands.

### 5. ~~Push the branch~~ — DONE

`zn/focused-mclaren-c3e069` is pushed to `origin` (now 25 commits
ahead of `origin/main`; new-branch ref set to track
`origin/zn/focused-mclaren-c3e069`). No PR has been opened — that's
up to whoever reviews the stack. GitHub printed the canonical
`pull/new/zn/focused-mclaren-c3e069` URL on push if it's wanted.

## Standalone repo (`~/hf/modelweightvis`)

### 6. ~~Update the `arbvis` path-dep~~ — DONE

Standalone `~/hf/modelweightvis` commit `dbe1e57` mirrors arbvis
`5e71cd6a` (SourceMetaSidecarHook) + `37a3e5d9` (ModelOpts CLI
split) and flips the `Cargo.toml` path-dep from the worktree
location to `../arbvis/crates/arbvis`. Cargo.lock untouched (same
arbvis 0.1.0 crate, different on-disk path). cargo fmt + clippy
(zero warnings) + `cargo test --lib` (87 tests) clean; release
build succeeds; local smoke of SmolLM2-135M `--output` produces a
valid 4096² PNG.

### 7. Tag arbvis v0.1.0

Original plan: tag `arbvis v0.1.0` in `~/hf/arbvis` after the split
lands. The standalone modelweightvis can then drop the `path =`
override entirely and pin against the published version.

The arbvis surface is stable enough to tag — every type
`modelweightvis` reaches into is `pub`, and the trait surface
(`FormatPlugin`, `LayoutPlugin`, `DiffSourceBuilder`,
`MoeDiffPrep` / `RepoDiffPrep` / `DirectoryTensorDiffPrep` /
`FinetuneDetect` / `SingleImageArchHook`) is the agreed contract.

## Probably not worth chasing

- **Revisit `?Send` on the four hook traits.** Today they use
  `#[async_trait(?Send)]` because hf-hub's `snapshot_download_impl`
  contains closures with implicit lifetimes the rustc HRTB check
  can't resolve. If hf-hub fixes this upstream, the `+ Send` bound
  can come back. No behaviour change for users either way.

- **Drop the per-item `#[allow(dead_code)]` markers.** Three in
  modelweightvis (the sidecar cluster, `color_for_pos`, and
  `ModelInfo.color_ranges`) each have an explanatory comment.
  They'll come off automatically when (2) lands and when a future
  Hilbert+dtype overlay consumes `color_ranges`.
