use std::path::Path;

use anyhow::{bail, Context};

use crate::hf_cli;
use crate::hf_url::{self, HfOutputSpec, RepoKind};
use crate::throttle::with_throttle;

/// Upload a local directory tree to a path prefix in a Hugging Face repo.
pub async fn upload_dir(
    local_dir: &Path,
    repo_id: &str,
    kind: RepoKind,
    path_prefix: &str,
) -> anyhow::Result<()> {
    log::info!(
        "Uploading {}/ → hf://{}/{} ...",
        local_dir.display(),
        repo_id,
        path_prefix
    );
    let local_str = local_dir.to_string_lossy().into_owned();
    let label = format!("hf upload-dir {repo_id}/{path_prefix}");

    match kind {
        RepoKind::Bucket => {
            let dest = if path_prefix.is_empty() {
                format!("hf://buckets/{repo_id}")
            } else {
                format!("hf://buckets/{repo_id}/{path_prefix}")
            };
            with_throttle(&label, || async {
                // `hf sync <local> <hf://...> --delete` matches the prior
                // `BucketSyncDirection::Upload` + `delete=true` semantics.
                hf_cli::run_hf(["sync", local_str.as_str(), dest.as_str(), "--delete"]).await
            })
            .await
            .with_context(|| format!("syncing {} -> {dest}", local_dir.display()))?;
        }
        kind => {
            let repo_type = kind.cli_repo_type()?;
            // `hf upload` defaults PATH_IN_REPO to the local relative path; we
            // mirror the prior explicit-prefix behavior by passing "." when
            // the prefix is empty.
            let prefix = if path_prefix.is_empty() {
                "."
            } else {
                path_prefix
            };
            with_throttle(&label, || async {
                hf_cli::run_hf([
                    "upload",
                    "--repo-type",
                    repo_type,
                    repo_id,
                    local_str.as_str(),
                    prefix,
                ])
                .await
            })
            .await
            .with_context(|| {
                format!(
                    "uploading dir {} -> hf://{repo_id}/{path_prefix}",
                    local_dir.display()
                )
            })?;
        }
    }
    Ok(())
}

/// Parse an `hf://` output URL and upload a local directory tree to the target repo.
pub async fn upload_dir_to(hf_url_str: &str, local_dir: &Path) -> anyhow::Result<()> {
    let spec = hf_url::parse_hf_output(hf_url_str)?;
    upload_dir(local_dir, &spec.repo_id, spec.kind, &spec.path_prefix).await
}

/// Create (or verify) the HF bucket that backs a Space, and return an
/// `HfOutputSpec` pointing at it. The bucket id is `<namespace>/<repo>_bucket`.
pub async fn create_space_bucket(space_id: &str) -> anyhow::Result<HfOutputSpec> {
    let bucket_id = derive_bucket_id(space_id)?;
    log::info!("Ensuring bucket {} exists...", bucket_id);
    with_throttle(&format!("hf buckets create {bucket_id}"), || async {
        hf_cli::run_hf(["buckets", "create", bucket_id.as_str(), "--exist-ok"]).await
    })
    .await
    .with_context(|| format!("creating bucket {bucket_id}"))?;
    Ok(HfOutputSpec {
        repo_id: bucket_id,
        kind: RepoKind::Bucket,
        revision: "main".to_string(),
        path_prefix: String::new(),
    })
}

/// Deploy (or redeploy) the Space app files — app.py, README, Dockerfile,
/// requirements.txt, and index.html. Assumes tiles and labels.json are already
/// in the bucket (either synced from local disk or streamed directly).
pub async fn deploy_space_app(
    space_id: &str,
    bucket_id: &str,
    index_html: Vec<u8>,
) -> anyhow::Result<()> {
    // Probe before create so we know whether this command is creating the
    // Space fresh (no restart needed — it builds and starts on its own) or
    // redeploying onto an existing one (restart below to pick up new content).
    let pre_existing = space_exists(space_id).await?;

    log::info!("Ensuring Space {} exists...", space_id);
    with_throttle(&format!("hf repos create {space_id}"), || async {
        hf_cli::run_hf([
            "repos",
            "create",
            "--type",
            "space",
            "--space-sdk",
            "docker",
            "--exist-ok",
            space_id,
        ])
        .await
    })
    .await
    .with_context(|| format!("creating space {space_id}"))?;

    log::info!("Uploading Space files...");
    let tmp = tempfile::TempDir::new()?;
    write_space_files(tmp.path(), bucket_id, space_id)?;
    std::fs::write(tmp.path().join("index.html"), index_html)?;

    let local_str = tmp.path().to_string_lossy().into_owned();
    with_throttle(&format!("hf upload {space_id} (space files)"), || async {
        hf_cli::run_hf([
            "upload",
            "--repo-type",
            "space",
            space_id,
            local_str.as_str(),
            ".",
        ])
        .await
    })
    .await
    .with_context(|| format!("uploading Space files to {space_id}"))?;

    log::info!("Deployed to https://huggingface.co/spaces/{}", space_id);

    // An `hf upload` that changes no app files won't trigger a Hub rebuild, so a
    // redeploy onto an existing Space can keep serving stale bucket content.
    // Restart it explicitly. Best-effort: the upload already succeeded, so a
    // failed control-plane restart is a warning, not a deploy failure.
    if pre_existing {
        log::info!("Restarting existing Space {}...", space_id);
        if let Err(e) = restart_space(space_id).await {
            log::warn!(
                "Deployed, but failed to restart Space {space_id}: {e:#}. \
                 Restart it manually from the Space's Settings if it serves stale content."
            );
        }
    }

    Ok(())
}

/// Whether a Space repo already exists. Mirrors `hf_url::fetch_model_card`'s
/// direct-reqwest pattern. A `404` is a clean "does not exist"; `2xx` and the
/// gated `401`/`403` both mean it exists (we just may lack read access).
async fn space_exists(space_id: &str) -> anyhow::Result<bool> {
    let url = format!("{}/api/spaces/{space_id}", hf_url::endpoint());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("building reqwest client")?;
    let mut req = client.get(&url);
    if let Some(tok) = hf_url::read_token() {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await.context("HF space_info request failed")?;
    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        Ok(false)
    } else if status.is_success()
        || status == reqwest::StatusCode::UNAUTHORIZED
        || status == reqwest::StatusCode::FORBIDDEN
    {
        Ok(true)
    } else {
        bail!("HF space_info for {space_id} returned unexpected status {status}");
    }
}

/// Trigger a normal restart of a Space (not a factory reboot — that would wipe
/// persistent storage). Matches `HfApi.restart_space`: `POST /api/spaces/{id}/restart`.
async fn restart_space(space_id: &str) -> anyhow::Result<()> {
    let url = format!("{}/api/spaces/{space_id}/restart", hf_url::endpoint());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("building reqwest client")?;
    let mut req = client.post(&url);
    if let Some(tok) = hf_url::read_token() {
        req = req.bearer_auth(tok);
    }
    req.send()
        .await
        .context("HF restart request failed")?
        .error_for_status()
        .context("HF restart returned non-2xx status")?;
    Ok(())
}

pub async fn run_deploy(tiles_dir: &Path, space_id: &str) -> anyhow::Result<()> {
    validate_tiles_dir(tiles_dir)?;

    let bucket_spec = create_space_bucket(space_id).await?;
    let bucket_id = &bucket_spec.repo_id;

    log::info!("Syncing tiles to bucket (this may take a while for large outputs)...");
    let tiles_path = tiles_dir.join("tiles");
    let tiles_local = tiles_path.to_string_lossy().into_owned();
    let tiles_dest = format!("hf://buckets/{bucket_id}/tiles");
    with_throttle(&format!("hf sync tiles -> {bucket_id}"), || async {
        hf_cli::run_hf([
            "sync",
            tiles_local.as_str(),
            tiles_dest.as_str(),
            "--delete",
        ])
        .await
    })
    .await
    .with_context(|| format!("syncing tiles to bucket {bucket_id}"))?;

    log::info!("Uploading labels.json to bucket...");
    let labels_path = tiles_dir.join("labels.json");
    let labels_local = labels_path.to_string_lossy().into_owned();
    let labels_dest = format!("hf://buckets/{bucket_id}/labels.json");
    with_throttle(
        &format!("hf buckets cp labels.json -> {bucket_id}"),
        || async {
            hf_cli::run_hf(["buckets", "cp", labels_local.as_str(), labels_dest.as_str()]).await
        },
    )
    .await
    .with_context(|| format!("uploading labels.json to bucket {bucket_id}"))?;

    let index_html = std::fs::read(tiles_dir.join("index.html"))
        .context("failed to read index.html from tiles directory")?;
    deploy_space_app(space_id, bucket_id, index_html).await
}

/// Deploy a 3D (`--3d`) bundle directory to a Space: sync the whole bundle
/// (`index.html`, `volume.bin`, `bricks.bin`, `pagetable.bin`, `meta.json`)
/// into the backing bucket and (re)deploy the Space app that serves it. The
/// generalized `app.py` static catch-all serves the assets from the bucket.
pub async fn run_deploy_bundle(dir: &Path, space_id: &str) -> anyhow::Result<()> {
    let index = dir.join("index.html");
    if !index.exists() {
        bail!("index.html not found in {}", dir.display());
    }

    let bucket_spec = create_space_bucket(space_id).await?;
    let bucket_id = &bucket_spec.repo_id;

    log::info!("Syncing 3D bundle to bucket (this may take a while)...");
    upload_dir(dir, bucket_id, RepoKind::Bucket, "").await?;

    let index_html = std::fs::read(&index).context("failed to read index.html from bundle")?;
    deploy_space_app(space_id, bucket_id, index_html).await
}

fn validate_tiles_dir(dir: &Path) -> anyhow::Result<()> {
    for name in ["index.html", "labels.json"] {
        let p = dir.join(name);
        if !p.exists() {
            bail!("{} not found in {}", name, dir.display());
        }
    }
    let tiles = dir.join("tiles");
    if !tiles.is_dir() {
        bail!("tiles/ directory not found in {}", dir.display());
    }
    Ok(())
}

fn derive_bucket_id(space_id: &str) -> anyhow::Result<String> {
    let (namespace, repo) = hf_url::split_owner_name(space_id)
        .with_context(|| format!("--space must be namespace/repo, got {space_id:?}"))?;
    if repo.ends_with("_bucket") {
        bail!(
            "--space repo name {repo:?} already ends in '_bucket'; arbvis appends '_bucket' \
             to derive the storage bucket and refuses to double-suffix it. \
             Pass a Space name without the '_bucket' suffix."
        );
    }
    Ok(format!("{namespace}/{repo}_bucket"))
}

fn write_space_files(dir: &Path, bucket_id: &str, space_id: &str) -> anyhow::Result<()> {
    let repo_name = space_id.split('/').nth(1).unwrap_or(space_id);

    let readme = include_str!("space_template/README.md.tmpl").replace("__REPO_NAME__", repo_name);
    std::fs::write(dir.join("README.md"), readme)?;

    std::fs::write(
        dir.join("Dockerfile"),
        include_str!("space_template/Dockerfile"),
    )?;

    std::fs::write(
        dir.join("requirements.txt"),
        include_str!("space_template/requirements.txt"),
    )?;

    let app_py = include_str!("space_template/app.py.tmpl").replace("__BUCKET_ID__", bucket_id);
    std::fs::write(dir.join("app.py"), app_py)?;

    Ok(())
}
