//! Thin subprocess wrapper around the official Python `hf` CLI.
//!
//! Every Hub I/O operation other than direct-HTTP byte ranges (which stay in
//! `xet/mod.rs` and `hf_url::fetch_range`) goes through here. The CLI bundles
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
    /// Path relative to the repo root.
    pub path: String,
    /// Serialized file size in bytes, omitted by the API for directories.
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    /// Git blob SHA-256 of the file, when the API reports it.
    pub blob_id: Option<String>,
    #[serde(default)]
    /// Xet chunk hash backing the file, when storage is xet-backed.
    pub xet_hash: Option<String>,
    #[serde(default)]
    /// Present when the file is stored via LFS or xet (see [`HfTreeLfs`]).
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

/// The `lfs` block of an [`HfTreeEntry`] for LFS- or xet-backed files.
#[derive(Debug, Clone, Deserialize)]
pub struct HfTreeLfs {
    #[serde(default)]
    /// SHA-256 of the LFS object contents.
    pub sha256: Option<String>,
    #[serde(default)]
    /// Size of the stored object in bytes.
    pub size: Option<u64>,
    #[serde(default)]
    /// Size of the LFS pointer file itself, when applicable.
    pub pointer_size: Option<u64>,
}

/// Subprocess failure modes. Maps onto the AIMD throttle outcomes via
/// `impl ErrorClassify` below.
#[derive(Debug)]
pub enum HfCliError {
    Spawn {
        bin: String,
        source: std::io::Error,
    },
    /// Reading the child's stdout failed mid-stream; the captured output is
    /// truncated, so it must not be parsed or acted on.
    StdoutRead(std::io::Error),
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
    TimedOut {
        argv: String,
        seconds: u64,
    },
}

impl std::fmt::Display for HfCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfCliError::Spawn { bin, source } => {
                if bin == "hf" {
                    write!(
                        f,
                        "failed to spawn `hf` CLI ({source}). Install with `pip install -U huggingface_hub` or `brew install huggingface-cli`."
                    )
                } else {
                    write!(
                        f,
                        "failed to spawn `{bin}` (the `hf` CLI, set via ARBVIS_HF_BIN) ({source}). Check that ARBVIS_HF_BIN names an executable on $PATH."
                    )
                }
            }
            HfCliError::Exit {
                argv,
                status,
                stderr_excerpt,
            } => {
                write!(f, "`hf {argv}` exited {status}: {stderr_excerpt}")
            }
            HfCliError::JsonDecode {
                argv,
                stderr_excerpt,
                source,
            } => {
                write!(
                    f,
                    "decoding `hf {argv}` JSON output failed: {source}\nstderr tail: {stderr_excerpt}"
                )
            }
            HfCliError::StdoutRead(e) => write!(
                f,
                "reading `hf` CLI output failed mid-stream ({e}); captured stdout is incomplete"
            ),
            HfCliError::TimedOut { argv, seconds } => {
                write!(
                    f,
                    "`hf {argv}` did not exit within {seconds}s (ARBVIS_HF_TIMEOUT_SECS) and was killed"
                )
            }
        }
    }
}

impl std::error::Error for HfCliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HfCliError::Spawn { source, .. } => Some(source),
            HfCliError::StdoutRead(e) => Some(e),
            HfCliError::Exit { .. } => None,
            HfCliError::JsonDecode { source, .. } => Some(source),
            HfCliError::TimedOut { .. } => None,
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
            HfCliError::Spawn { .. } => Outcome::Permanent,
            // Truncated output: retrying won't heal a failed pipe read, and
            // acting on partial output would be silently wrong.
            HfCliError::StdoutRead(_) => Outcome::Permanent,
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
            // A killed child is indistinguishable from a network stall from
            // the caller's point of view — let the retry budget decide.
            HfCliError::TimedOut { .. } => Outcome::Timeout,
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

/// Overall wall-clock budget for a single `hf` invocation, opt-in via
/// `ARBVIS_HF_TIMEOUT_SECS`. Unset or non-numeric means no timeout: uploads
/// of large checkpoints can legitimately run for hours, so arbvis never
/// kills a healthy child by default. When set, a child that exceeds the
/// budget is killed and the call fails loudly (classify → `Timeout`, so the
/// AIMD throttle treats it like any other stall).
fn hf_timeout_secs() -> Option<u64> {
    match std::env::var("ARBVIS_HF_TIMEOUT_SECS") {
        Ok(s) => parse_hf_timeout(&s),
        Err(_) => None,
    }
}

/// Parse one `ARBVIS_HF_TIMEOUT_SECS` value. Anything that is not a positive
/// integer (including `0`, which would make the timeout fire instantly) means
/// no timeout; a warning is logged for every ignored value so a typo'd or
/// zero setting never silently disables the opt-in budget.
fn parse_hf_timeout(s: &str) -> Option<u64> {
    match s.trim().parse::<u64>() {
        Ok(secs) if secs > 0 => Some(secs),
        _ => {
            log::warn!(
                "ARBVIS_HF_TIMEOUT_SECS={s:?} is not a positive integer; ignoring (no timeout)"
            );
            None
        }
    }
}

/// Wrap an I/O failure from spawning (or waiting on) the configured `hf`
/// binary, recording which binary was involved so the message can name it.
fn spawn_err(source: std::io::Error) -> HfCliError {
    HfCliError::Spawn {
        bin: hf_binary(),
        source,
    }
}

/// Build the `Exit` error for a failed `hf` invocation, first logging the
/// stderr excerpt at debug level so failures are diagnosable without
/// surfacing them in normal output.
fn exit_error(argv_display: String, status: ExitStatus, stderr_excerpt: String) -> HfCliError {
    log::debug!("`hf {argv_display}` failed with {status}; stderr: {stderr_excerpt}");
    HfCliError::Exit {
        argv: argv_display,
        status,
        stderr_excerpt,
    }
}

/// Spawn `hf <args>` and return (exit status, captured stdout, captured stderr tail).
///
/// stderr is forwarded to the parent process's stderr line-by-line as it
/// arrives, AND a bounded tail is kept for the error excerpt. stdout is
/// captured in full (it carries `--json` payloads).
/// Read `reader` to end, propagating read errors so a failed capture is
/// never mistaken for complete output. Tested with a reader that fails
/// mid-stream (see `failing_read_propagates_error`).
async fn read_all<R>(mut reader: R) -> std::io::Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).await?;
    Ok(buf)
}

async fn run_and_capture<I, S>(args: I) -> Result<(ExitStatus, Vec<u8>, String), HfCliError>
where
    I: IntoIterator<Item = S> + Clone,
    S: AsRef<OsStr>,
{
    let mut cmd = build_cmd(args.clone());
    let mut child = cmd.spawn().map_err(spawn_err)?;

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
    // A read error is propagated (not swallowed): truncated output must fail
    // loudly rather than be parsed as if complete.
    let stdout_task = tokio::spawn(read_all(stdout));

    let timeout_secs = hf_timeout_secs();
    let status = match timeout_secs {
        Some(secs) => {
            let budget = std::time::Duration::from_secs(secs);
            match tokio::time::timeout(budget, child.wait()).await {
                Ok(status) => status.map_err(spawn_err)?,
                Err(_) => {
                    // kill_on_drop(true) reaps on scope exit, but kill
                    // explicitly so the pipe readers below see EOF now.
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(HfCliError::TimedOut {
                        argv: argv_for_display(args),
                        seconds: secs,
                    });
                }
            }
        }
        None => child.wait().await.map_err(spawn_err)?,
    };

    let stdout = match stdout_task.await {
        Ok(Ok(buf)) => buf,
        Ok(Err(e)) => return Err(HfCliError::StdoutRead(e)),
        Err(join) => {
            let e = std::io::Error::other(format!("stdout capture task failed: {join}"));
            return Err(HfCliError::StdoutRead(e));
        }
    };
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
        return Err(exit_error(argv_display, status, stderr_excerpt));
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
        return Err(exit_error(argv_display, status, stderr_excerpt));
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
        return Err(exit_error(argv_display, status, stderr_excerpt));
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
        return Err(exit_error("--version".to_string(), status, stderr_excerpt));
    }
    Ok(String::from_utf8_lossy(&stdout).trim().to_string())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Reader that serves a few bytes, then fails: simulates a pipe error
    /// mid-capture of the child's stdout.
    struct FailingReader {
        yielded: usize,
    }

    impl tokio::io::AsyncRead for FailingReader {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            if self.yielded == 0 {
                self.yielded += 1;
                buf.put_slice(b"partial");
                return std::task::Poll::Ready(Ok(()));
            }
            std::task::Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "injected mid-stream failure",
            )))
        }
    }

    #[tokio::test]
    async fn read_all_propagates_mid_stream_read_error_instead_of_silently_truncating() {
        let err = read_all(FailingReader { yielded: 0 })
            .await
            .expect_err("a mid-stream read failure must not be swallowed into truncated output");
        assert_eq!(err.kind(), std::io::ErrorKind::BrokenPipe);
    }

    #[tokio::test]
    async fn read_all_collects_full_output_on_success() {
        let buf = read_all(&b"{\"ok\": 1}"[..]).await.unwrap();
        assert_eq!(buf, b"{\"ok\": 1}");
    }

    #[test]
    fn parse_hf_timeout_values() {
        assert_eq!(parse_hf_timeout("30"), Some(30));
        assert_eq!(parse_hf_timeout(" 30 "), Some(30));
        // 0 would make the opt-in budget fire instantly; treat it like any
        // other non-positive value: warn and run without a timeout.
        assert_eq!(parse_hf_timeout("0"), None);
        assert_eq!(parse_hf_timeout("abc"), None);
        assert_eq!(parse_hf_timeout(""), None);
        assert_eq!(parse_hf_timeout("-1"), None);
    }

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
    fn classify_timed_out_is_timeout() {
        let err = HfCliError::TimedOut {
            argv: "download foo/bar".into(),
            seconds: 30,
        };
        assert_eq!(err.classify(), Outcome::Timeout);
    }

    /// Injects a hung child: ARBVIS_HF_BIN points at a script that never
    /// exits. Without the ARBVIS_HF_TIMEOUT_SECS kill this test would hang
    /// until the harness timeout; with it, the call must fail loudly with
    /// `TimedOut` well before the child's own sleep elapses.
    #[tokio::test]
    async fn hung_child_is_killed_at_optin_timeout() {
        let _env = ENV_LOCK.lock().await;
        let mut path = std::env::temp_dir();
        path.push(format!("arbvis-hf-hang-{}.sh", std::process::id()));
        std::fs::write(&path, "#!/bin/sh\nsleep 300\n").expect("write fake hf script");
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod fake hf script");
        }
        let prev_bin = std::env::var("ARBVIS_HF_BIN").ok();
        let prev_timeout = std::env::var("ARBVIS_HF_TIMEOUT_SECS").ok();
        std::env::set_var("ARBVIS_HF_BIN", &path);
        std::env::set_var("ARBVIS_HF_TIMEOUT_SECS", "1");

        let start = std::time::Instant::now();
        let result = check_hf_available().await;

        // Restore env before assertions so failures don't poison siblings.
        match (prev_bin, prev_timeout) {
            (Some(b), Some(t)) => {
                std::env::set_var("ARBVIS_HF_BIN", b);
                std::env::set_var("ARBVIS_HF_TIMEOUT_SECS", t);
            }
            (Some(b), None) => {
                std::env::set_var("ARBVIS_HF_BIN", b);
                std::env::remove_var("ARBVIS_HF_TIMEOUT_SECS");
            }
            (None, Some(t)) => {
                std::env::remove_var("ARBVIS_HF_BIN");
                std::env::set_var("ARBVIS_HF_TIMEOUT_SECS", t);
            }
            (None, None) => {
                std::env::remove_var("ARBVIS_HF_BIN");
                std::env::remove_var("ARBVIS_HF_TIMEOUT_SECS");
            }
        }
        let _ = std::fs::remove_file(&path);

        let err = result.expect_err("hung child must fail, not hang");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(60),
            "timeout did not fire promptly: {:?}",
            start.elapsed()
        );
        assert!(matches!(err, HfCliError::TimedOut { seconds: 1, .. }));
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
        let err = HfCliError::Spawn {
            bin: "hf".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        assert_eq!(err.classify(), Outcome::Permanent);
    }

    #[test]
    fn spawn_display_suggests_install_for_default_binary() {
        let err = HfCliError::Spawn {
            bin: "hf".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        let msg = err.to_string();
        assert!(msg.contains("`hf` CLI"), "unexpected message: {msg}");
        assert!(msg.contains("pip install"), "unexpected message: {msg}");
    }

    #[test]
    fn spawn_display_names_custom_binary_from_arbvis_hf_bin() {
        let err = HfCliError::Spawn {
            bin: "/opt/tools/my-hf".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("`/opt/tools/my-hf`"),
            "unexpected message: {msg}"
        );
        assert!(msg.contains("ARBVIS_HF_BIN"), "unexpected message: {msg}");
        assert!(!msg.contains("pip install"), "unexpected message: {msg}");
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

    // The fake-binary tests mutate the process-global ARBVIS_HF_BIN env var;
    // cargo runs tests on parallel threads, so they serialize on this lock.
    pub(crate) static ENV_LOCK: tokio::sync::Mutex<()> =
        tokio::sync::Mutex::const_new(());

    /// Writes a fake `hf` executable to a temp path, points `ARBVIS_HF_BIN`
    /// at it, and restores the previous env value on drop so failures don't
    /// poison sibling tests.
    pub(crate) struct FakeHfBinGuard {
        path: std::path::PathBuf,
        prev: Option<String>,
    }

    impl FakeHfBinGuard {
        pub(crate) fn with_script(script: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "arbvis-hf-fake-{}-{}.sh",
                std::process::id(),
                uuid_like_suffix()
            ));
            std::fs::write(&path, script).expect("write fake hf script");
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod fake hf script");
            let prev = std::env::var("ARBVIS_HF_BIN").ok();
            std::env::set_var("ARBVIS_HF_BIN", &path);
            Self { path, prev }
        }
    }

    impl Drop for FakeHfBinGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(b) => std::env::set_var("ARBVIS_HF_BIN", b),
                None => std::env::remove_var("ARBVIS_HF_BIN"),
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }

    // Distinguishes concurrently-alive temp scripts from each other.
    pub(crate) fn uuid_like_suffix() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    /// The captured stderr tail is bounded at STDERR_CAPTURE_LIMIT and keeps
    /// the newest bytes: a child that emits far more than the cap on stderr
    /// must still surface its final lines in the error excerpt.
    #[tokio::test]
    async fn stderr_tail_is_bounded_and_keeps_the_newest_bytes() {
        let _env = ENV_LOCK.lock().await;
        let _guard = FakeHfBinGuard::with_script(
            "#!/bin/sh\nyes | head -c 10240 >&2\nprintf 'TAIL-MARKER-9Z\\n' >&2\nexit 3\n",
        );
        let err = check_hf_available()
            .await
            .expect_err("child exits non-zero");
        let HfCliError::Exit { stderr_excerpt, .. } = err else {
            panic!("expected Exit error, got {err:?}");
        };
        assert!(
            stderr_excerpt.len() <= STDERR_CAPTURE_LIMIT,
            "excerpt {} exceeds cap",
            stderr_excerpt.len()
        );
        assert!(
            stderr_excerpt.ends_with("TAIL-MARKER-9Z\n"),
            "newest stderr line must survive the sliding window"
        );
    }

    /// `download` appends `--quiet` and returns the last non-empty stdout
    /// line (incidental leading output must not corrupt the path).
    #[tokio::test]
    async fn download_returns_last_nonempty_stdout_line() {
        let _env = ENV_LOCK.lock().await;
        let _guard = FakeHfBinGuard::with_script(
            "#!/bin/sh\nprintf 'progress noise\\n\\n/tmp/arbvis-test-snapshot\\n'\n",
        );
        let path = download(["download", "foo/bar"])
            .await
            .expect("download succeeds");
        assert_eq!(path, std::path::PathBuf::from("/tmp/arbvis-test-snapshot"));
    }

    /// A zero-exit `download` that prints no path is an error, not a silent
    /// empty path.
    #[tokio::test]
    async fn download_reports_error_when_stdout_has_no_path() {
        let _env = ENV_LOCK.lock().await;
        let _guard = FakeHfBinGuard::with_script("#!/bin/sh\nexit 0\n");
        let err = download(["download", "foo/bar"])
            .await
            .expect_err("no path printed");
        let HfCliError::Exit { stderr_excerpt, .. } = err else {
            panic!("expected Exit error, got {err:?}");
        };
        assert!(stderr_excerpt.contains("printed no path"));
    }

    /// `run_hf_json` appends `--json` to the argv it passes to the child and
    /// decodes the child's stdout as the requested type.
    #[tokio::test]
    async fn run_hf_json_appends_json_flag_and_decodes() {
        let _env = ENV_LOCK.lock().await;
        let _guard = FakeHfBinGuard::with_script(
            "#!/bin/sh\nif [ \"$1\" = buckets ] && [ \"$2\" = ls ] && [ \"$3\" = -R ] && [ \"$4\" = --json ]; then\n  printf '[{\\\"path\\\":\\\"a.bin\\\",\\\"size\\\":5}]'\nelse\n  printf 'unexpected argv: %s\\n' \"$*\" >&2\n  exit 9\nfi\n",
        );
        let entries: Vec<HfTreeEntry> = run_hf_json(["buckets", "ls", "-R"])
            .await
            .expect("listing decodes");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "a.bin");
        assert_eq!(entries[0].size, Some(5));
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
