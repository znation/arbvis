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

### 4. End-to-end smoke verification for remote scenarios

**Status.** Verified locally: arbvis `/bin/ls --output`, arbvis
`/tmp/big.bin --tiles`, modelweightvis local-safetensors `--tiles`,
arbvis JSON `--diff --output`. All byte-identical or deterministic.

**Gap.** The plan's remote scenarios are unverified:
- `modelweightvis hf://meta-llama/Llama-3.2-1B --tiles /tmp/mw`
- `modelweightvis --diff hf://meta-llama/Llama-3.2-1B hf://meta-llama/Llama-3.2-1B-Instruct --tiles /tmp/mwd`
- `modelweightvis --moe-diff hf://mistralai/Mixtral-8x7B-v0.1 --tiles /tmp/mwmoe`
- Directory `--diff <dir> <dir>` with a mix of tensor + non-tensor
  files

The hooks for each are wired, but the end-to-end paths haven't been
exercised post-relocation. The repo-diff and moe-diff paths in
particular shuttle a lot of state through `materialize_remote_arcs`
(which I restored to its real impl in commit `4c0e7be`) and the
`TensorMoeDiffPrep` / `TensorRepoDiffPrep` / `TensorDirectoryDiffPrep`
hooks.

**Fix.** Run each scenario against the network and compare output to
a pre-relocation baseline (which exists in `/tmp/arbvis-step12e` for
some of these). Pixel-diff with `compare` or a python script;
acceptable thresholds documented in the plan's "explicitly out of
scope" note (any unexplained pixel diff is a bug).

**Why deferred.** Network access + remote-state mutability make this
hard to fully automate. Worth a manual pass before any release tag.

### 5. Push the branch

`zn/focused-mclaren-c3e069` is 20 commits ahead of `origin/main`. No
push has been done. The branch is well-structured and reviewable as
a stack.

## Standalone repo (`~/hf/modelweightvis`)

### 6. Update the `arbvis` path-dep

`Cargo.toml` pins `arbvis` to the in-progress worktree:
```toml
arbvis = { path = "../arbvis/.claude/worktrees/focused-mclaren-c3e069/crates/arbvis", version = "0.1.0" }
```

Once `zn/focused-mclaren-c3e069` lands on `~/hf/arbvis`'s `main`,
update to:
```toml
arbvis = { path = "../arbvis/crates/arbvis", version = "0.1.0" }
```

The `version = "0.1.0"` pin already matches the workspace; only the
path changes.

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
