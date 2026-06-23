//! Thin subprocess wrapper around the official Python `hf` CLI.
//!
//! Every Hub I/O operation other than direct-HTTP byte ranges (which stay in
//! `xet.rs` and `hf_url::fetch_range`) goes through here. The CLI bundles
//! `hf-xet` internally, so whole-file downloads keep xet dedup wire-speedup
//! for free without arbvis pulling in xet's per-call stream-group rebuild.
//!
//! Conventions:
//! - `download` appends `--quiet` and returns the local path `hf download`
//!   prints to stdout. The `--json` output mode was removed in
//!   huggingface_hub ≥ 1.0; `--quiet` suppresses progress bars and prints
//!   only the resulting path (the file when a filename is given, else the
//!   snapshot dir).
//! - `run_hf_json` appends `--json` and parses stdout as `T`. Used for the
//!   `hf buckets ls -R` listing (`[ HfTreeEntry... ]`); model/dataset/space
//!   listings now go through the Hub tree API in `hf_url`.
//! - `run_hf` ignores stdout. Used for `upload`, `upload-large-folder`,
//!   `sync`, `buckets cp`, and the `repos create` / `buckets create`
//!   bootstrap calls.
//! - stdout is captured. stderr is teed: forwarded to the parent terminal
//!   (so the user sees the CLI's native progress bars) AND captured into a
//!   bounded ring so the last few KB are available for the error excerpt
//!   when the process exits non-zero.
//! - `HF_TOKEN` is set from `hf_url::read_token()` when present; otherwise
//!   the env is inherited so `hf` falls back to `~/.cache/huggingface/token`
//!   on its own.

use std::ffi::OsStr;
use std::io::Write;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::hf_url;
use crate::throttle::{ErrorClassify, Outcome};

/// Cap on captured stderr per invocation. Big enough to hold a Python
/// traceback worth diagnosing; small enough not to balloon `anyhow` chains
/// when many subprocesses fail in a retry storm.
const STDERR_CAPTURE_LIMIT: usize = 4 * 1024;

/// Entry in a recursive repo file listing — `hf buckets ls -R --json` for
/// buckets, or the Hub tree API (see `hf_url`) for model/dataset/space repos.
/// Directories surface with `size = None`; only files have `blob_id` /
/// `xet_hash` / `lfs`. Unknown fields are ignored by serde's default.
#[derive(Debug, Clone, Deserialize)]
pub struct HfTreeEntry {
    pub path: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub blob_id: Option<String>,
    #[serde(default)]
    pub xet_hash: Option<String>,
    #[serde(default)]
    pub lfs: Option<HfTreeLfs>,
}

impl HfTreeEntry {
    /// True when this entry refers to a file (not a directory). Buckets and
    /// repos both omit `size` for directory entries.
    #[inline]
    pub fn is_file(&self) -> bool {
        self.size.is_some()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfTreeLfs {
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub pointer_size: Option<u64>,
}

/// Subprocess failure modes. Maps onto the AIMD throttle outcomes via
/// `impl ErrorClassify` below.
#[derive(Debug)]
pub enum HfCliError {
    Spawn(std::io::Error),
    Exit {
        argv: String,
        status: ExitStatus,
        stderr_excerpt: String,
    },
    JsonDecode {
        argv: String,
        stderr_excerpt: String,
        source: serde_json::Error,
    },
}

impl std::fmt::Display for HfCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfCliError::Spawn(e) => write!(
                f,
                "failed to spawn `hf` CLI ({e}). Install with `pip install -U huggingface_hub` or `brew install huggingface-cli`."
            ),
            HfCliError::Exit { argv, status, stderr_excerpt } => {
                write!(f, "`hf {argv}` exited {status}: {stderr_excerpt}")
            }
            HfCliError::JsonDecode { argv, stderr_excerpt, source } => {
                write!(
                    f,
                    "decoding `hf {argv}` JSON output failed: {source}\nstderr tail: {stderr_excerpt}"
                )
            }
        }
    }
}

impl std::error::Error for HfCliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HfCliError::Spawn(e) => Some(e),
            HfCliError::Exit { .. } => None,
            HfCliError::JsonDecode { source, .. } => Some(source),
        }
    }
}

impl ErrorClassify for HfCliError {
    /// Best-effort classification from stderr text. The CLI doesn't expose
    /// structured error types, so we substring-match its messages and log
    /// the full text at `debug!` on every non-zero exit so misclassifications
    /// are diagnosable from logs.
    fn classify(&self) -> Outcome {
        match self {
            // Missing binary won't fix itself — don't burn the AIMD retry budget on it.
            HfCliError::Spawn(_) => Outcome::Permanent,
            HfCliError::JsonDecode { .. } => Outcome::Permanent,
            HfCliError::Exit { stderr_excerpt, .. } => {
                let s = stderr_excerpt.to_ascii_lowercase();
                if s.contains("429")
                    || s.contains("rate limit")
                    || s.contains("rate-limit")
                    || s.contains("too many requests")
                {
                    Outcome::RateLimit
                } else if s.contains("timeout")
                    || s.contains("timed out")
                    || s.contains("connection reset")
                    || s.contains("connection refused")
                    || s.contains("connection error")
                    || s.contains("temporarily unavailable")
                    || s.contains(" 500 ")
                    || s.contains(" 502 ")
                    || s.contains(" 503 ")
                    || s.contains(" 504 ")
                {
                    Outcome::Timeout
                } else {
                    Outcome::Permanent
                }
            }
        }
    }
}

/// Name of the `hf` binary on `$PATH`. Overridable via `ARBVIS_HF_BIN` for
/// tests or for users with the CLI under a non-standard name.
fn hf_binary() -> String {
    std::env::var("ARBVIS_HF_BIN").unwrap_or_else(|_| "hf".to_string())
}

/// Format the argv for inclusion in error messages.
fn argv_for_display<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    args.into_iter()
        .map(|a| a.as_ref().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build a `tokio::process::Command` for the given `hf` args, with token /
/// stderr handling wired up. Caller adds `--json` etc. as needed.
fn build_cmd<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new(hf_binary());
    cmd.args(args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped()); // we tee in our own task
    cmd.kill_on_drop(true);

    // Pass the token explicitly when arbvis has one resolved. When we don't,
    // `hf` falls back to its own `~/.cache/huggingface/token` lookup, which
    // is the right behavior for users who've already run `hf auth login`.
    if let Some(token) = hf_url::read_token() {
        cmd.env("HF_TOKEN", token);
    }
    // Disable the CLI's "you're behind by N versions" stderr nag inside
    // arbvis runs; users can update on their own time.
    cmd.env("HF_HUB_DISABLE_UPDATE_CHECK", "1");

    cmd
}

/// Spawn `hf <args>` and return (exit status, captured stdout, captured stderr tail).
///
/// stderr is forwarded to the parent process's stderr line-by-line as it
/// arrives, AND a bounded tail is kept for the error excerpt. stdout is
/// captured in full (it carries `--json` payloads).
async fn run_and_capture<I, S>(args: I) -> Result<(ExitStatus, Vec<u8>, String), HfCliError>
where
    I: IntoIterator<Item = S> + Clone,
    S: AsRef<OsStr>,
{
    let mut cmd = build_cmd(args);
    let mut child = cmd.spawn().map_err(HfCliError::Spawn)?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let stderr_buf = Arc::new(Mutex::new(Vec::<u8>::with_capacity(STDERR_CAPTURE_LIMIT)));
    let stderr_buf_clone = stderr_buf.clone();
    let stderr_task = tokio::spawn(async move {
        let mut reader = stderr;
        let mut chunk = [0u8; 4096];
        loop {
            match reader.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => {
                    // Forward to the user so they see the CLI's native progress.
                    let _ = std::io::stderr().write_all(&chunk[..n]);
                    // Keep a bounded tail for error excerpts.
                    let mut buf = stderr_buf_clone.lock().await;
                    if buf.len() + n > STDERR_CAPTURE_LIMIT {
                        // Slide window: drop oldest bytes to make room.
                        let overflow = (buf.len() + n).saturating_sub(STDERR_CAPTURE_LIMIT);
                        if overflow >= buf.len() {
                            buf.clear();
                        } else {
                            buf.drain(..overflow);
                        }
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                Err(_) => break,
            }
        }
    });

    // Read all of stdout into a buffer (no size cap — JSON payloads can be
    // large for repo listings, and truncating mid-array breaks the parser).
    let stdout_task = tokio::spawn(async move {
        let mut reader = stdout;
        let mut buf = Vec::new();
        let _ = reader.read_to_end(&mut buf).await;
        buf
    });

    let status = child.wait().await.map_err(HfCliError::Spawn)?;

    let stdout = stdout_task.await.unwrap_or_default();
    let _ = stderr_task.await;
    let stderr_tail = {
        let buf = stderr_buf.lock().await;
        String::from_utf8_lossy(&buf).into_owned()
    };

    Ok((status, stdout, stderr_tail))
}

/// Run `hf <args>` and return success/failure. Used for upload / sync /
/// create calls where we don't parse output.
pub async fn run_hf<I, S>(args: I) -> Result<(), HfCliError>
where
    I: IntoIterator<Item = S> + Clone,
    S: AsRef<OsStr>,
{
    let argv_display = argv_for_display(args.clone());
    let (status, _stdout, stderr_excerpt) = run_and_capture(args).await?;
    if !status.success() {
        log::debug!("`hf {argv_display}` failed with {status}; stderr: {stderr_excerpt}");
        return Err(HfCliError::Exit {
            argv: argv_display,
            status,
            stderr_excerpt,
        });
    }
    Ok(())
}

/// Run `hf <args> --json` and parse stdout as `T`.
///
/// The caller is responsible for ensuring `--json` is meaningful for the
/// subcommand. `--json` is appended automatically.
pub async fn run_hf_json<T, I, S>(args: I) -> Result<T, HfCliError>
where
    T: DeserializeOwned,
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut argv: Vec<std::ffi::OsString> = args
        .into_iter()
        .map(|s| s.as_ref().to_os_string())
        .collect();
    argv.push("--json".into());
    let argv_display = argv_for_display(argv.iter());

    let (status, stdout, stderr_excerpt) = run_and_capture(argv).await?;
    if !status.success() {
        log::debug!("`hf {argv_display}` failed with {status}; stderr: {stderr_excerpt}");
        return Err(HfCliError::Exit {
            argv: argv_display,
            status,
            stderr_excerpt,
        });
    }

    serde_json::from_slice::<T>(&stdout).map_err(|source| HfCliError::JsonDecode {
        argv: argv_display,
        stderr_excerpt,
        source,
    })
}

/// Run `hf download <args> --quiet` and return the local path it printed.
///
/// `hf download` no longer has a `--json` mode (removed in huggingface_hub
/// ≥ 1.0). `--quiet` disables progress bars and prints only the resulting
/// local path to stdout — the file path when a filename is given, otherwise
/// the snapshot directory. We take the last non-empty stdout line so any
/// incidental leading output doesn't corrupt the path.
///
/// `--quiet` is appended automatically; the caller passes the rest of the
/// `download …` argv.
pub async fn download<I, S>(args: I) -> Result<std::path::PathBuf, HfCliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut argv: Vec<std::ffi::OsString> = args
        .into_iter()
        .map(|s| s.as_ref().to_os_string())
        .collect();
    argv.push("--quiet".into());
    let argv_display = argv_for_display(argv.iter());

    let (status, stdout, stderr_excerpt) = run_and_capture(argv).await?;
    if !status.success() {
        log::debug!("`hf {argv_display}` failed with {status}; stderr: {stderr_excerpt}");
        return Err(HfCliError::Exit {
            argv: argv_display,
            status,
            stderr_excerpt,
        });
    }

    let text = String::from_utf8_lossy(&stdout);
    let path = text
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(str::trim)
        .unwrap_or("");
    if path.is_empty() {
        return Err(HfCliError::Exit {
            argv: argv_display,
            status,
            stderr_excerpt: format!(
                "`hf download --quiet` exited 0 but printed no path; stderr tail: {stderr_excerpt}"
            ),
        });
    }
    Ok(std::path::PathBuf::from(path))
}

/// One-shot probe: run `hf --version` and return the trimmed version line.
/// Callers should invoke this before the first hub op so a missing CLI
/// fails fast with a clear install hint.
pub async fn check_hf_available() -> Result<String, HfCliError> {
    let (status, stdout, stderr_excerpt) = run_and_capture(["--version"]).await?;
    if !status.success() {
        return Err(HfCliError::Exit {
            argv: "--version".to_string(),
            status,
            stderr_excerpt,
        });
    }
    Ok(String::from_utf8_lossy(&stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_rate_limit() {
        let err = HfCliError::Exit {
            argv: "download foo/bar".into(),
            status: ExitStatus::default(),
            stderr_excerpt: "HTTPError: 429 Too Many Requests".into(),
        };
        assert_eq!(err.classify(), Outcome::RateLimit);
    }

    #[test]
    fn classify_timeout() {
        let err = HfCliError::Exit {
            argv: "download foo/bar".into(),
            status: ExitStatus::default(),
            stderr_excerpt: "ConnectionError: Connection reset by peer".into(),
        };
        assert_eq!(err.classify(), Outcome::Timeout);
    }

    #[test]
    fn classify_5xx_status_line() {
        let err = HfCliError::Exit {
            argv: "download foo/bar".into(),
            status: ExitStatus::default(),
            stderr_excerpt: "Server returned 503 Service Unavailable".into(),
        };
        // Substring " 503 " requires a leading + trailing space.
        assert_eq!(err.classify(), Outcome::Timeout);
    }

    #[test]
    fn classify_permanent_default() {
        let err = HfCliError::Exit {
            argv: "upload foo/bar".into(),
            status: ExitStatus::default(),
            stderr_excerpt: "RepositoryNotFoundError: repo 'foo/bar' not found".into(),
        };
        assert_eq!(err.classify(), Outcome::Permanent);
    }

    #[test]
    fn classify_spawn_is_permanent() {
        let err = HfCliError::Spawn(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert_eq!(err.classify(), Outcome::Permanent);
    }

    #[test]
    fn tree_entry_directory_has_no_size() {
        let entry: HfTreeEntry = serde_json::from_str(r#"{"path": "subdir"}"#).unwrap();
        assert!(!entry.is_file());
    }

    #[test]
    fn tree_entry_file_has_size() {
        let entry: HfTreeEntry =
            serde_json::from_str(r#"{"path": "config.json", "size": 665}"#).unwrap();
        assert!(entry.is_file());
        assert_eq!(entry.size, Some(665));
    }

    /// Smoke test against the real `hf` CLI. Ignored by default so cargo
    /// test doesn't require it on PATH; run with `cargo test -p arbvis
    /// -- --ignored hf_cli::tests::smoke_real_cli_version` when you need
    /// to confirm wiring after a build.
    #[tokio::test]
    #[ignore = "requires `hf` on PATH"]
    async fn smoke_real_cli_version() {
        let version = check_hf_available().await.expect("hf --version");
        assert!(!version.is_empty(), "expected non-empty version line");
        eprintln!("hf --version -> {version}");
    }
}
