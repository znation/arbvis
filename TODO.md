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

### 2. Wire `SourceMeta` sidecar enrichment

**Status.** [`crates/modelweightvis/src/data.rs`](crates/modelweightvis/src/data.rs)
has `try_load_source_meta`, `load_meta_for_sources`, and
`fetch_hf_sidecar` defined, each with `#[allow(dead_code)]` and a
comment explaining the gap. `ArchLayoutPlugin::build` passes `&[]` for
sidecars to `ArchLayout::try_build`.

**Gap.** Transformer-aware layout grouping (read `config.json`'s
`num_hidden_layers` + `hidden_size`, read
`model.safetensors.index.json` for shard stitching) doesn't run. The
arch layout works without it — falls back to inferring structure from
tensor names — but is less faithful for unusual model layouts.

**Fix.** Two options:
- Add a new arbvis hook (`PrepareSourcesExtension` or similar) that
  runs after `prepare_sources`, takes `&mut [Source]`, and lets
  modelweightvis call `load_meta_for_sources` and merge results into
  each source's `extensions`. Then `ArchLayoutPlugin::build` reads
  `source.extensions.get::<SourceMeta>()` and threads it into
  `ArchLayout::try_build`.
- Or extend `FormatPlugin` with a `sibling_probe(parent_dir: &Path) ->
  Option<SourceMeta>` method that the local-populate path calls. Loses
  the cross-source dedup that `load_meta_for_sources` does, so the
  first option is cleaner.

**Why deferred.** Arch layout is correct without it; just less rich.

### 3. Move model-side CLI flags off `arbvis::Args`

**Status.** The original plan ([§Step 12 in `tender-streamed-kitten.md`](.claude/plans/using-claude-plans-ok-i-think-the-tender-streamed-kitten.md))
called for `modelweightvis::Args` to clap-flatten `arbvis::Args` and
add `--moe-diff`, `--finetune`/`--no-finetune`, `--diff-metric`,
`--layout`. Today these flags all live on `arbvis::Args` in
[`crates/arbvis/src/lib.rs`](crates/arbvis/src/lib.rs:150).

**Gap.** Architectural rough edge — the byte-only `arbvis` CLI
accepts model-side flags but errors at runtime when the corresponding
hook isn't registered ("--moe-diff requires a tensor-aware backend").
A user reading `arbvis --help` sees flags that don't work without
modelweightvis.

**Fix.** Create `modelweightvis::Args` that clap-flattens
`arbvis::Args` and owns the four model-side flags. Demote the same
four flags from `arbvis::Args`. `arbvis::run` takes both args halves
(or just the flattened struct, which works because clap can derive
across crates).

**Why deferred.** Cosmetic in the help text; flags are functional
today. Real fix requires non-trivial clap reshaping.

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
