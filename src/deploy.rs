use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use hf_hub::buckets::sync::BucketSyncDirection;
use hf_hub::buckets::BucketUpload;
use hf_hub::repository::AddSource;
use hf_hub::{HFClientSync, RepoTypeDataset, RepoTypeModel, RepoTypeSpace};

use crate::hf_url::HfOutputSpec;

/// Upload a single local file to a Hugging Face repo via hf-hub.
///
/// `repo_type` must be one of `"model"`, `"dataset"`, `"space"`, or `"bucket"`.
pub fn upload_file(
    local: &Path,
    repo_id: &str,
    repo_type: &str,
    path_in_repo: &str,
) -> anyhow::Result<()> {
    log::info!("Uploading {} → hf://{}/{} ...", local.display(), repo_id, path_in_repo);
    let client = crate::hf_url::client()?;
    let (owner, name) = split_owner_name(repo_id)?;
    let source = AddSource::file(PathBuf::from(local));
    let dest = path_in_repo.to_string();

    match repo_type {
        "model" => {
            client.model(owner, name)
                .upload_file().source(source).path_in_repo(dest).send()?;
        }
        "dataset" => {
            client.dataset(owner, name)
                .upload_file().source(source).path_in_repo(dest).send()?;
        }
        "space" => {
            client.space(owner, name)
                .upload_file().source(source).path_in_repo(dest).send()?;
        }
        "bucket" => {
            client.bucket(owner, name)
                .upload_files()
                .files(vec![BucketUpload::new(PathBuf::from(local), dest)])
                .send()?;
        }
        other => bail!("unknown repo type {other:?}"),
    }
    Ok(())
}

/// Upload a local directory tree to a path prefix in a Hugging Face repo.
pub fn upload_dir(
    local_dir: &Path,
    repo_id: &str,
    repo_type: &str,
    path_prefix: &str,
) -> anyhow::Result<()> {
    log::info!("Uploading {}/ → hf://{}/{} ...", local_dir.display(), repo_id, path_prefix);
    let client = crate::hf_url::client()?;
    let (owner, name) = split_owner_name(repo_id)?;
    let folder = PathBuf::from(local_dir);
    let prefix_opt = if path_prefix.is_empty() { None } else { Some(path_prefix.to_string()) };

    match repo_type {
        "model" => upload_folder_typed::<RepoTypeModel>(&client, owner, name, folder, prefix_opt)?,
        "dataset" => upload_folder_typed::<RepoTypeDataset>(&client, owner, name, folder, prefix_opt)?,
        "space" => upload_folder_typed::<RepoTypeSpace>(&client, owner, name, folder, prefix_opt)?,
        "bucket" => {
            let prefix = if path_prefix.is_empty() { None } else { Some(path_prefix.to_string()) };
            client.bucket(owner, name)
                .sync()
                .local_path(folder)
                .direction(BucketSyncDirection::Upload)
                .maybe_prefix(prefix)
                .send()?;
        }
        other => bail!("unknown repo type {other:?}"),
    }
    Ok(())
}

fn upload_folder_typed<T: hf_hub::RepoType>(
    client: &HFClientSync,
    owner: &str,
    name: &str,
    folder: PathBuf,
    path_in_repo: Option<String>,
) -> anyhow::Result<()>
where
    HFClientSync: TypedRepoFactory<T>,
{
    let repo = <HFClientSync as TypedRepoFactory<T>>::repo(client, owner, name);
    repo.upload_folder()
        .folder_path(folder)
        .maybe_path_in_repo(path_in_repo)
        .send()?;
    Ok(())
}

trait TypedRepoFactory<T: hf_hub::RepoType> {
    fn repo(client: &Self, owner: &str, name: &str) -> hf_hub::HFRepositorySync<T>;
}
impl TypedRepoFactory<RepoTypeModel> for HFClientSync {
    fn repo(client: &Self, owner: &str, name: &str) -> hf_hub::HFRepositorySync<RepoTypeModel> {
        client.model(owner, name)
    }
}
impl TypedRepoFactory<RepoTypeDataset> for HFClientSync {
    fn repo(client: &Self, owner: &str, name: &str) -> hf_hub::HFRepositorySync<RepoTypeDataset> {
        client.dataset(owner, name)
    }
}
impl TypedRepoFactory<RepoTypeSpace> for HFClientSync {
    fn repo(client: &Self, owner: &str, name: &str) -> hf_hub::HFRepositorySync<RepoTypeSpace> {
        client.space(owner, name)
    }
}

/// Create (or verify) the HF bucket that backs a Space, and return an
/// `HfOutputSpec` pointing at it.
pub fn create_space_bucket(space_id: &str) -> anyhow::Result<(HfOutputSpec, String)> {
    let bucket_id = derive_bucket_id(space_id)?;
    eprintln!("Ensuring bucket {} exists...", bucket_id);
    let client = crate::hf_url::client()?;
    let (owner, name) = split_owner_name(&bucket_id)?;
    client
        .create_bucket()
        .namespace(owner.to_string())
        .name(name.to_string())
        .exist_ok(true)
        .send()
        .with_context(|| format!("creating bucket {bucket_id}"))?;
    let spec = HfOutputSpec {
        repo_id: bucket_id.clone(),
        repo_type_str: "bucket",
        revision: "main".to_string(),
        path_prefix: String::new(),
    };
    Ok((spec, bucket_id))
}

/// Deploy (or redeploy) the Space app files — app.py, README, Dockerfile,
/// requirements.txt, and index.html. Assumes tiles and labels.json are already
/// in the bucket (either synced from local disk or streamed directly).
pub fn deploy_space_app(space_id: &str, bucket_id: &str, index_html: Vec<u8>) -> anyhow::Result<()> {
    eprintln!("Ensuring Space {} exists...", space_id);
    let client = crate::hf_url::client()?;
    client
        .create_repository()
        .repo_id(space_id.to_string())
        .repo_type(RepoTypeSpace)
        .space_sdk("docker")
        .exist_ok(true)
        .send()
        .with_context(|| format!("creating space {space_id}"))?;

    eprintln!("Uploading Space files...");
    let tmp = tempfile::TempDir::new()?;
    write_space_files(tmp.path(), bucket_id, space_id)?;
    std::fs::write(tmp.path().join("index.html"), index_html)?;

    let (owner, name) = split_owner_name(space_id)?;
    client
        .space(owner, name)
        .upload_folder()
        .folder_path(tmp.path().to_path_buf())
        .send()
        .with_context(|| format!("uploading Space files to {space_id}"))?;

    eprintln!("Deployed to https://huggingface.co/spaces/{}", space_id);
    Ok(())
}

pub fn run_deploy(tiles_dir: &Path, space_id: &str) -> anyhow::Result<()> {
    validate_tiles_dir(tiles_dir)?;

    let (_, bucket_id) = create_space_bucket(space_id)?;
    let client = crate::hf_url::client()?;
    let (owner, name) = split_owner_name(&bucket_id)?;
    let bucket = client.bucket(owner, name);

    eprintln!("Syncing tiles to bucket (this may take a while for large outputs)...");
    let tiles_path = tiles_dir.join("tiles");
    bucket
        .sync()
        .local_path(tiles_path)
        .direction(BucketSyncDirection::Upload)
        .prefix("tiles".to_string())
        .delete(true)
        .send()
        .with_context(|| format!("syncing tiles to bucket {bucket_id}"))?;

    eprintln!("Uploading labels.json to bucket...");
    let labels_path = tiles_dir.join("labels.json");
    bucket
        .upload_files()
        .files(vec![BucketUpload::new(labels_path, "labels.json".to_string())])
        .send()
        .with_context(|| format!("uploading labels.json to bucket {bucket_id}"))?;

    let index_html = std::fs::read(tiles_dir.join("index.html"))
        .context("failed to read index.html from tiles directory")?;
    deploy_space_app(space_id, &bucket_id, index_html)
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
    let (namespace, repo) = split_owner_name(space_id)
        .with_context(|| format!("--space must be namespace/repo, got {space_id:?}"))?;
    Ok(format!("{}/{}_bucket", namespace, repo))
}

fn split_owner_name(repo_id: &str) -> anyhow::Result<(&str, &str)> {
    let slash = repo_id
        .find('/')
        .with_context(|| format!("expected owner/name, got {repo_id:?}"))?;
    Ok((&repo_id[..slash], &repo_id[slash + 1..]))
}

fn write_space_files(dir: &Path, bucket_id: &str, space_id: &str) -> anyhow::Result<()> {
    let repo_name = space_id.split('/').nth(1).unwrap_or(space_id);
    let readme = format!(
        "---\ntitle: \"arbvis: {repo_name}\"\nemoji: 📊\ncolorFrom: blue\ncolorTo: indigo\nsdk: docker\napp_port: 7860\npinned: false\n---\n"
    );
    std::fs::write(dir.join("README.md"), readme)?;

    std::fs::write(
        dir.join("Dockerfile"),
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 7860\nCMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"7860\"]\n",
    )?;

    std::fs::write(
        dir.join("requirements.txt"),
        "fastapi\nuvicorn[standard]\nhuggingface_hub\n",
    )?;

    let app_py = format!(
        r#"import os
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from huggingface_hub import HfFileSystem

BUCKET_ID = "{bucket_id}"

app = FastAPI()
_fs = None


def fs():
    global _fs
    if _fs is None:
        _fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    return _fs


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/labels.json")
def labels():
    data = fs().read_bytes(f"hf://buckets/{{BUCKET_ID}}/labels.json")
    return Response(content=data, media_type="application/json")


@app.get("/tiles/{{z}}/{{x}}/{{y_png}}")
def tile(z: int, x: int, y_png: str):
    path = f"hf://buckets/{{BUCKET_ID}}/tiles/{{z}}/{{x}}/{{y_png}}"
    try:
        data = fs().read_bytes(path)
        return Response(content=data, media_type="image/png")
    except Exception:
        return Response(status_code=404)
"#,
        bucket_id = bucket_id,
    );
    std::fs::write(dir.join("app.py"), app_py)?;

    Ok(())
}
