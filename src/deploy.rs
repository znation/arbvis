use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use hf_hub::buckets::sync::BucketSyncDirection;
use hf_hub::buckets::BucketUpload;
use hf_hub::repository::AddSource;
use hf_hub::RepoTypeSpace;

use crate::hf_url::{self, HfOutputSpec, RepoKind};
use crate::throttle::with_throttle;

/// Upload a single local file to a Hugging Face repo via hf-hub.
pub async fn upload_file(
    local: &Path,
    repo_id: &str,
    kind: RepoKind,
    path_in_repo: &str,
) -> anyhow::Result<()> {
    log::info!(
        "Uploading {} → hf://{}/{} ...",
        local.display(),
        repo_id,
        path_in_repo
    );
    let client = hf_url::client()?;
    let (owner, name) = hf_url::split_owner_name(repo_id)?;
    let source = AddSource::file(PathBuf::from(local));
    let dest = path_in_repo.to_string();

    let label = format!("upload_file {repo_id}/{path_in_repo}");
    with_throttle(&label, || async {
        match kind {
            RepoKind::Model => client
                .model(owner, name)
                .upload_file()
                .source(source.clone())
                .path_in_repo(dest.clone())
                .send()
                .await
                .map(|_| ()),
            RepoKind::Dataset => client
                .dataset(owner, name)
                .upload_file()
                .source(source.clone())
                .path_in_repo(dest.clone())
                .send()
                .await
                .map(|_| ()),
            RepoKind::Space => client
                .space(owner, name)
                .upload_file()
                .source(source.clone())
                .path_in_repo(dest.clone())
                .send()
                .await
                .map(|_| ()),
            RepoKind::Bucket => client
                .bucket(owner, name)
                .upload_files()
                .files(vec![BucketUpload::new(PathBuf::from(local), dest.clone())])
                .send()
                .await
                .map(|_| ()),
        }
    })
    .await?;
    Ok(())
}

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
    let client = hf_url::client()?;
    let (owner, name) = hf_url::split_owner_name(repo_id)?;
    let folder = PathBuf::from(local_dir);
    let prefix = if path_prefix.is_empty() {
        None
    } else {
        Some(path_prefix.to_string())
    };

    let label = format!("upload_dir {repo_id}/{path_prefix}");
    with_throttle(&label, || async {
        match kind {
            RepoKind::Model => client
                .model(owner, name)
                .upload_folder()
                .folder_path(folder.clone())
                .maybe_path_in_repo(prefix.clone())
                .send()
                .await
                .map(|_| ()),
            RepoKind::Dataset => client
                .dataset(owner, name)
                .upload_folder()
                .folder_path(folder.clone())
                .maybe_path_in_repo(prefix.clone())
                .send()
                .await
                .map(|_| ()),
            RepoKind::Space => client
                .space(owner, name)
                .upload_folder()
                .folder_path(folder.clone())
                .maybe_path_in_repo(prefix.clone())
                .send()
                .await
                .map(|_| ()),
            RepoKind::Bucket => client
                .bucket(owner, name)
                .sync()
                .local_path(folder.clone())
                .direction(BucketSyncDirection::Upload)
                .maybe_prefix(prefix.clone())
                .send()
                .await
                .map(|_| ()),
        }
    })
    .await?;
    Ok(())
}

/// Parse an `hf://` output URL and upload a single local file to the target repo.
pub async fn upload_file_to(hf_url_str: &str, local: &Path) -> anyhow::Result<()> {
    let spec = hf_url::parse_hf_output(hf_url_str)?;
    upload_file(local, &spec.repo_id, spec.kind, &spec.path_prefix).await
}

/// Parse an `hf://` output URL and upload a local directory tree to the target repo.
pub async fn upload_dir_to(hf_url_str: &str, local_dir: &Path) -> anyhow::Result<()> {
    let spec = hf_url::parse_hf_output(hf_url_str)?;
    upload_dir(local_dir, &spec.repo_id, spec.kind, &spec.path_prefix).await
}

/// Create (or verify) the HF bucket that backs a Space, and return an
/// `HfOutputSpec` pointing at it. The bucket id is `spec.repo_id`.
pub async fn create_space_bucket(space_id: &str) -> anyhow::Result<HfOutputSpec> {
    let bucket_id = derive_bucket_id(space_id)?;
    log::info!("Ensuring bucket {} exists...", bucket_id);
    let client = hf_url::client()?;
    let (owner, name) = hf_url::split_owner_name(&bucket_id)?;
    with_throttle(&format!("create_bucket {bucket_id}"), || async {
        client
            .create_bucket()
            .namespace(owner.to_string())
            .name(name.to_string())
            .exist_ok(true)
            .send()
            .await
            .map(|_| ())
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
    log::info!("Ensuring Space {} exists...", space_id);
    let client = hf_url::client()?;
    with_throttle(&format!("create_repository {space_id}"), || async {
        client
            .create_repository()
            .repo_id(space_id.to_string())
            .repo_type(RepoTypeSpace)
            .space_sdk("docker")
            .exist_ok(true)
            .send()
            .await
            .map(|_| ())
    })
    .await
    .with_context(|| format!("creating space {space_id}"))?;

    log::info!("Uploading Space files...");
    let tmp = tempfile::TempDir::new()?;
    write_space_files(tmp.path(), bucket_id, space_id)?;
    std::fs::write(tmp.path().join("index.html"), index_html)?;

    let (owner, name) = hf_url::split_owner_name(space_id)?;
    let folder = tmp.path().to_path_buf();
    with_throttle(&format!("space upload_folder {space_id}"), || async {
        client
            .space(owner, name)
            .upload_folder()
            .folder_path(folder.clone())
            .send()
            .await
            .map(|_| ())
    })
    .await
    .with_context(|| format!("uploading Space files to {space_id}"))?;

    log::info!("Deployed to https://huggingface.co/spaces/{}", space_id);
    Ok(())
}

pub async fn run_deploy(tiles_dir: &Path, space_id: &str) -> anyhow::Result<()> {
    validate_tiles_dir(tiles_dir)?;

    let bucket_spec = create_space_bucket(space_id).await?;
    let bucket_id = &bucket_spec.repo_id;
    let client = hf_url::client()?;
    let (owner, name) = hf_url::split_owner_name(bucket_id)?;
    let bucket = client.bucket(owner, name);

    log::info!("Syncing tiles to bucket (this may take a while for large outputs)...");
    let tiles_path = tiles_dir.join("tiles");
    with_throttle(&format!("bucket sync {bucket_id}"), || async {
        bucket
            .sync()
            .local_path(tiles_path.clone())
            .direction(BucketSyncDirection::Upload)
            .prefix("tiles".to_string())
            .delete(true)
            .send()
            .await
            .map(|_| ())
    })
    .await
    .with_context(|| format!("syncing tiles to bucket {bucket_id}"))?;

    log::info!("Uploading labels.json to bucket...");
    let labels_path = tiles_dir.join("labels.json");
    with_throttle(&format!("bucket upload labels {bucket_id}"), || async {
        bucket
            .upload_files()
            .files(vec![BucketUpload::new(
                labels_path.clone(),
                "labels.json".to_string(),
            )])
            .send()
            .await
            .map(|_| ())
    })
    .await
    .with_context(|| format!("uploading labels.json to bucket {bucket_id}"))?;

    let index_html = std::fs::read(tiles_dir.join("index.html"))
        .context("failed to read index.html from tiles directory")?;
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
