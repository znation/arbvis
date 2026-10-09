//! Shared `indicatif` style helpers.
//!
//! Three styles cover the patterns used across the crate:
//! - [`counter_style`]: a left-anchored bar tracking total work done, with rate
//!   and ETA suffix (`{per_sec}, ETA {eta}`).
//! - [`queue_style`]: an indented secondary bar for instantaneous queue fill,
//!   anchored under whatever counter it follows.
//! - [`status_style`]: no bar — just a status message and the elapsed time.
//!   Used for the AIMD throttle indicator.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use indicatif::style::ProgressTracker;
use indicatif::{HumanDuration, MultiProgress, ProgressDrawTarget, ProgressState, ProgressStyle};

/// Threshold above which the rate-based ETA is suppressed.
///
/// Early in a run (or after a stall) the smoothed rate can be near zero, which
/// makes the ETA blow up to weeks or months even for jobs that finish in
/// minutes. Past this cap we first try a lifetime-average ETA (`(len - pos) *
/// elapsed / pos`, prefixed with `~`) and fall back to `--` only if even that
/// exceeds the cap.
const ETA_DISPLAY_CAP: Duration = Duration::from_secs(24 * 60 * 60);

/// Half-life for the smart_eta EWMA, in seconds.
///
/// Indicatif's built-in estimator uses a hardcoded 15s window, which makes
/// ETA over-react to recent stalls or bursts (a 30s stall decays the rate by
/// 100×). We feed the same `(pos, instant)` samples through a longer-window
/// estimator so smart_eta stabilizes faster than that. `per_sec` keeps the
/// short-window default — it's the right tool for spotting current slowness.
const SMART_ETA_HALF_LIFE_SECS: f64 = 60.0;

/// Process-wide `MultiProgress` that owns every progress bar across the run.
///
/// All bars (whether grouped under a `PipelineProgress`, standalone counters,
/// or upload trackers) attach themselves to this `MultiProgress` so they
/// share a single draw target and renderer. Log output is routed through it
/// via `indicatif-log-bridge` (see `LogWrapper` in `main`), so any `log::*`
/// call automatically suspends bar rendering, prints, and resumes — instead
/// of overwriting bar lines.
///
/// The draw target is stderr; if stderr isn't a TTY, indicatif's hidden
/// target is used so non-interactive runs stay log-only.
pub fn multi() -> &'static MultiProgress {
    static M: OnceLock<MultiProgress> = OnceLock::new();
    M.get_or_init(|| {
        use std::io::IsTerminal;
        if std::io::stderr().is_terminal() {
            MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
        } else {
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
        }
    })
}

/// Progress-bar style for byte-position counters (bar with rate and ETA).
pub fn counter_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg} ({per_sec}, ETA {smart_eta})",
    )
    .expect("counter_style template parse")
    .progress_chars("=>-")
    .with_key("smart_eta", SmartEta::new(SMART_ETA_HALF_LIFE_SECS))
}

/// `ProgressTracker` driving the `smart_eta` template key.
///
/// Maintains its own single-EWMA `(pos, instant)` estimator independent of
/// indicatif's built-in one, so we can pick a longer half-life. On every tick
/// (position update or steady-tick), the estimator advances using the same
/// continuous-time weighting formula indicatif uses internally:
/// `w = 0.1 ^ (Δt / half_life)`. The estimator is the only writer; `write`
/// reads the latest smoothed rate to render the ETA.
#[derive(Clone)]
struct SmartEta {
    inner: Arc<Mutex<SmartEtaInner>>,
    half_life_secs: f64,
}

#[derive(Default)]
struct SmartEtaInner {
    smoothed_per_sec: f64,
    last_sample: Option<(u64, Instant)>,
}

impl SmartEta {
    fn new(half_life_secs: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SmartEtaInner::default())),
            half_life_secs,
        }
    }
}

impl ProgressTracker for SmartEta {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, state: &ProgressState, now: Instant) {
        let pos = state.pos();
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((prev_pos, prev_t)) = s.last_sample {
            let dt = now.saturating_duration_since(prev_t).as_secs_f64();
            if dt > 0.0 && pos >= prev_pos {
                let instant_rate = (pos - prev_pos) as f64 / dt;
                let weight = 0.1_f64.powf(dt / self.half_life_secs);
                s.smoothed_per_sec = s.smoothed_per_sec * weight + instant_rate * (1.0 - weight);
            }
        }
        s.last_sample = Some((pos, now));
    }

    fn reset(&mut self, _: &ProgressState, _: Instant) {
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.smoothed_per_sec = 0.0;
        s.last_sample = None;
    }

    fn write(&self, state: &ProgressState, w: &mut dyn std::fmt::Write) {
        let rate = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .smoothed_per_sec;
        let pos = state.pos();
        let len = state.len().unwrap_or(0);

        if rate > 0.0 && len > pos {
            if let Some(eta) = duration_within_cap((len - pos) as f64 / rate) {
                let _ = write!(w, "{:#}", HumanDuration(eta));
                return;
            }
        }
        // Long-window estimator has collapsed (e.g. extended stall) or no
        // samples yet. Fall back to the lifetime-average rate before giving
        // up. `~` marks the fallback so the reader can tell it isn't reacting
        // to recent throughput like the adjacent `per_sec`.
        let elapsed = state.elapsed().as_secs_f64();
        if pos > 0 && len > pos && elapsed > 0.0 {
            let lifetime_rate = pos as f64 / elapsed;
            if lifetime_rate > 0.0 {
                if let Some(fallback) = duration_within_cap((len - pos) as f64 / lifetime_rate) {
                    let _ = write!(w, "~{:#}", HumanDuration(fallback));
                    return;
                }
            }
        }
        let _ = w.write_str("--");
    }
}

/// Build a `Duration` from `secs` only if it's finite and within
/// [`ETA_DISPLAY_CAP`]. Returns `None` otherwise.
///
/// `Duration::from_secs_f64` panics on NaN, negative, infinity, or values
/// that overflow `Duration`. Any of those slipping into the ETA formatter
/// poisons indicatif's internal locks (the panic unwinds while a draw lock
/// is held), which cascades into `PoisonError` panics from every other
/// thread touching a progress bar. Validate first, construct second.
fn duration_within_cap(secs: f64) -> Option<Duration> {
    if !secs.is_finite() || secs < 0.0 || secs > ETA_DISPLAY_CAP.as_secs_f64() {
        return None;
    }
    Some(Duration::from_secs_f64(secs))
}

/// Indented sub-bar style for per-queue progress under a main counter.
pub fn queue_style() -> ProgressStyle {
    ProgressStyle::with_template("  └─ {msg}: {pos}/{len} [{bar:30.yellow/dim}]")
        .expect("queue_style template parse")
        .progress_chars("=>-")
}

/// Plain message style for status lines that only show elapsed time.
pub fn status_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}  ({elapsed_precise})").expect("status_style template parse")
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::{ProgressBar, TermLike};
    use std::fmt::Debug;
    use std::sync::{Arc, Mutex};

    type IoResult = std::io::Result<()>;

    /// A fake terminal that records every rendered line so tests can assert
    /// on what `SmartEta::write` actually produced.
    #[derive(Default)]
    struct CapturingTerm {
        lines: Arc<Mutex<Vec<String>>>,
    }

    impl Clone for CapturingTerm {
        fn clone(&self) -> Self {
            CapturingTerm {
                lines: Arc::clone(&self.lines),
            }
        }
    }

    impl CapturingTerm {
        fn lines(&self) -> Vec<String> {
            self.lines.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }
    }

    impl Debug for CapturingTerm {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("CapturingTerm")
        }
    }

    impl TermLike for CapturingTerm {
        fn width(&self) -> u16 {
            200
        }
        fn move_cursor_up(&self, _: usize) -> IoResult {
            Ok(())
        }
        fn move_cursor_down(&self, _: usize) -> IoResult {
            Ok(())
        }
        fn move_cursor_right(&self, _: usize) -> IoResult {
            Ok(())
        }
        fn move_cursor_left(&self, _: usize) -> IoResult {
            Ok(())
        }
        fn write_line(&self, s: &str) -> IoResult {
            self.lines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(s.to_owned());
            Ok(())
        }
        fn write_str(&self, s: &str) -> IoResult {
            self.lines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(s.to_owned());
            Ok(())
        }
        fn clear_line(&self) -> IoResult {
            Ok(())
        }
        fn flush(&self) -> IoResult {
            Ok(())
        }
    }

    fn bar_with_capture() -> (ProgressBar, CapturingTerm) {
        let term = CapturingTerm::default();
        let pb = ProgressBar::with_draw_target(
            None,
            ProgressDrawTarget::term_like(Box::new(term.clone())),
        );
        pb.set_length(1000);
        pb.set_style(counter_style());
        (pb, term)
    }

    /// The most recent non-blank rendered line — indicatif redraws the same
    /// bar each tick, so this is the current render.
    fn last_render(term: &CapturingTerm) -> String {
        term.lines()
            .into_iter()
            .rev()
            .find(|l| !l.trim().is_empty())
            .expect("no lines rendered")
    }

    fn assert_rendered(term: &CapturingTerm, needle: &str, want: bool) {
        let text = last_render(term);
        assert_eq!(
            text.contains(needle),
            want,
            "expected {} in: {:?}",
            if want { "presence of" } else { "absence of" },
            text
        );
    }

    /// Assert that every ETA occurrence in the render is exactly the bare
    /// `--` placeholder — i.e. neither the smoothed-rate nor the lifetime
    /// fallback path produced a duration.
    fn assert_only_placeholder_eta(term: &CapturingTerm) {
        for tail in last_render(term).split("ETA ").skip(1) {
            assert!(
                tail.starts_with("--"),
                "expected bare `--` ETA, got: {:?}",
                tail
            );
        }
    }

    #[test]
    fn duration_within_cap_accepts_only_finite_in_range_values() {
        assert_eq!(duration_within_cap(60.0), Some(Duration::from_secs(60)));
        assert_eq!(duration_within_cap(0.0), Some(Duration::ZERO));
        // Exactly at the cap is still allowed.
        assert_eq!(
            duration_within_cap(ETA_DISPLAY_CAP.as_secs_f64()),
            Some(ETA_DISPLAY_CAP)
        );
        // Just past the cap is not.
        assert_eq!(
            duration_within_cap(ETA_DISPLAY_CAP.as_secs_f64() + 0.001),
            None
        );
        // NaN, negative, and infinity would each panic Duration::from_secs_f64.
        assert_eq!(duration_within_cap(f64::NAN), None);
        assert_eq!(duration_within_cap(-1.0), None);
        assert_eq!(duration_within_cap(f64::INFINITY), None);
        // Finite but overflows Duration's range.
        assert_eq!(duration_within_cap(1e30), None);
    }

    #[test]
    fn eta_renders_placeholder_before_any_samples() {
        let (pb, term) = bar_with_capture();
        pb.tick();
        // No samples yet: smart_eta has no rate to extrapolate from.
        assert_rendered(&term, "ETA --", true);
        pb.finish();
    }

    #[test]
    fn eta_falls_back_to_lifetime_estimate_marker() {
        let (pb, term) = bar_with_capture();
        pb.set_position(100);
        pb.tick();
        // A second sample at the same position (dt > 0, no progress) drives
        // the smoothed rate to zero, so the writer must use the
        // lifetime-average fallback, marked with `~`.
        std::thread::sleep(Duration::from_millis(60));
        pb.tick();
        assert_rendered(&term, "ETA ~", true);
        pb.finish();
    }

    #[test]
    fn eta_suppressed_when_pos_reaches_len() {
        let (pb, term) = bar_with_capture();
        pb.tick();
        std::thread::sleep(Duration::from_millis(60));
        pb.set_position(100);
        pb.tick();
        std::thread::sleep(Duration::from_millis(60));
        // Completed bars render only the bare `--` placeholder: write() never
        // emits a duration when len <= pos.
        pb.set_position(1000);
        pb.tick();
        assert_only_placeholder_eta(&term);
        pb.finish();
    }

    #[test]
    fn eta_recovers_after_reset() {
        let (pb, term) = bar_with_capture();
        pb.tick();
        std::thread::sleep(Duration::from_millis(60));
        pb.set_position(100);
        pb.tick();
        // Reset clears the estimator; the next render must go back to the
        // placeholder, not carry the stale rate forward.
        pb.reset();
        pb.set_position(50);
        std::thread::sleep(Duration::from_millis(60));
        pb.tick();
        assert_rendered(&term, "ETA ~", true);
        pb.finish();
    }
}
