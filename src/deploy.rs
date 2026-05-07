use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context};

pub fn run_deploy(tiles_dir: &Path, space_id: &str) -> anyhow::Result<()> {
    validate_tiles_dir(tiles_dir)?;

    let bucket_id = derive_bucket_id(space_id)?;

    eprintln!("Ensuring bucket {} exists...", bucket_id);
    hf_idempotent(&["buckets", "create", &bucket_id])?;

    eprintln!("Syncing tiles to bucket (this may take a while for large outputs)...");
    let tiles_path = tiles_dir.join("tiles");
    let bucket_tiles_url = format!("hf://buckets/{}/tiles", bucket_id);
    hf(&[
        "buckets",
        "sync",
        tiles_path.to_str().context("non-UTF-8 tiles path")?,
        &bucket_tiles_url,
        "--delete",
    ])?;

    eprintln!("Uploading labels.json to bucket...");
    let labels_path = tiles_dir.join("labels.json");
    let bucket_labels_url = format!("hf://buckets/{}/labels.json", bucket_id);
    hf(&[
        "buckets",
        "cp",
        labels_path.to_str().context("non-UTF-8 labels path")?,
        &bucket_labels_url,
    ])?;

    eprintln!("Ensuring Space {} exists...", space_id);
    hf_idempotent(&["repo", "create", space_id, "--repo-type=space"])?;

    eprintln!("Uploading Space files...");
    let tmp = tempfile::TempDir::new()?;
    write_space_files(tmp.path(), &bucket_id)?;

    for (src, dest) in [
        (tmp.path().join("README.md"), "README.md"),
        (tmp.path().join("Dockerfile"), "Dockerfile"),
        (tmp.path().join("requirements.txt"), "requirements.txt"),
        (tmp.path().join("app.py"), "app.py"),
    ] {
        hf(&[
            "upload",
            space_id,
            src.to_str().context("non-UTF-8 temp path")?,
            dest,
            "--repo-type=space",
        ])?;
    }

    let index_path = tiles_dir.join("index.html");
    hf(&[
        "upload",
        space_id,
        index_path.to_str().context("non-UTF-8 index path")?,
        "index.html",
        "--repo-type=space",
    ])?;

    eprintln!("Deployed to https://huggingface.co/spaces/{}", space_id);
    Ok(())
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
    let slash = space_id
        .find('/')
        .with_context(|| format!("--space must be namespace/repo, got {:?}", space_id))?;
    let namespace = &space_id[..slash];
    let repo = &space_id[slash + 1..];
    Ok(format!("{}/{}_bucket", namespace, repo))
}

fn hf(args: &[&str]) -> anyhow::Result<()> {
    let status = Command::new("hf")
        .args(args)
        .stdin(Stdio::null())
        .status()
        .context("failed to run `hf` — is the Hugging Face CLI installed?")?;
    if !status.success() {
        bail!("`hf {}` exited with {}", args.join(" "), status);
    }
    Ok(())
}

// Like hf() but ignores failures whose stderr mentions "already exist".
fn hf_idempotent(args: &[&str]) -> anyhow::Result<()> {
    let output = Command::new("hf")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .context("failed to run `hf` — is the Hugging Face CLI installed?")?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{}{}", stderr, stdout).to_lowercase();
    if combined.contains("already exist") || combined.contains("already exists") {
        return Ok(());
    }

    bail!(
        "`hf {}` exited with {}\nstderr: {}",
        args.join(" "),
        output.status,
        stderr
    );
}

fn write_space_files(dir: &Path, bucket_id: &str) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("README.md"),
        "---\ntitle: arbvis\nemoji: 🗺\ncolorFrom: blue\ncolorTo: indigo\nsdk: docker\napp_port: 7860\npinned: false\n---\n",
    )?;

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
