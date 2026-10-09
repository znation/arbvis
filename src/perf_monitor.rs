//! Periodic perf snapshot logger, opt-in via `ARBVIS_PERF_LOG=1`.
//!
//! Logs one line per second with:
//! - AIMD throttle state (in_flight / active_limit / max, backoff parkers,
//!   cumulative 429/timeout counts)
//! - Direct-CAS xet HTTP state (in_flight, completed/s, MB/s)
//!
//! Useful for locating pipeline stalls: a long run with `cas_in_flight > 0`
//! and `0 req/s` points at slow CAS reads; `cas_in_flight == 0` with
//! `throttle_backoff > 0` is AIMD parking; both zero points elsewhere
//! (CPU-bound parse, lock contention).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::throttle::Throttle;
use crate::xet;

/// Interpret `ARBVIS_PERF_LOG` and spawn the monitor if it is `1`. Returns a
/// shutdown handle the caller can drop to stop the task on exit. A set-but
/// unrecognized value logs a warning naming the accepted value instead of
/// being silently ignored.
pub fn spawn_if_enabled() -> Option<Arc<AtomicBool>> {
    match check_perf_log(std::env::var("ARBVIS_PERF_LOG").ok().as_deref()) {
        Ok(false) => return None,
        Err(msg) => {
            log::warn!("{msg}");
            return None;
        }
        Ok(true) => {}
    }
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_task = Arc::clone(&stop);
    tokio::spawn(async move {
        let mut last_tick = Instant::now();
        let mut last_cas_completed: u64 = 0;
        let mut last_cas_bytes: u64 = 0;
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if stop_for_task.load(Ordering::Relaxed) {
                break;
            }
            let now = Instant::now();
            let dt = now
                .saturating_duration_since(last_tick)
                .as_secs_f64()
                .max(0.001);
            last_tick = now;

            let ts = Throttle::global().stats();
            let cs = xet::cas_stats();
            let dcompleted = cs.completed.saturating_sub(last_cas_completed);
            let dbytes = cs.bytes.saturating_sub(last_cas_bytes);
            last_cas_completed = cs.completed;
            last_cas_bytes = cs.bytes;
            let req_per_s = dcompleted as f64 / dt;
            let mb_per_s = dbytes as f64 / dt / (1024.0 * 1024.0);

            log::info!(
                "perf: throttle in_flight={}/{} (max {}) backoff={} 429s={} timeouts={} | cas in_flight={} {:.1} req/s {:.1} MB/s",
                ts.in_flight, ts.active_limit, ts.max_workers, ts.in_backoff,
                ts.total_rate_limits, ts.total_timeouts,
                cs.in_flight, req_per_s, mb_per_s,
            );
        }
    });
    Some(stop)
}

/// Interpret `ARBVIS_PERF_LOG`: `Ok(true)` to spawn the monitor, `Ok(false)`
/// to stay quiet, `Err(msg)` for a set-but-unrecognized value — the caller
/// logs `msg` so a typo'd value is not silently ignored.
fn check_perf_log(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some("1") => Ok(true),
        // A set-but-unrecognized value means the user asked for perf logging
        // and would otherwise silently get nothing.
        Some(other) => Err(format!(
            "ignoring ARBVIS_PERF_LOG={other:?}: only `1` enables the perf monitor"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sets/unsets ARBVIS_PERF_LOG around `f` and restores the prior value.
    /// Tests touching this env var must hold ENV_LOCK (process-global state).
    fn with_perf_log(value: Option<&str>, f: impl FnOnce()) {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("ARBVIS_PERF_LOG").ok();
        match value {
            Some(v) => std::env::set_var("ARBVIS_PERF_LOG", v),
            None => std::env::remove_var("ARBVIS_PERF_LOG"),
        }
        f();
        match prev {
            Some(v) => std::env::set_var("ARBVIS_PERF_LOG", v),
            None => std::env::remove_var("ARBVIS_PERF_LOG"),
        }
    }

    #[test]
    fn disabled_when_env_unset() {
        with_perf_log(None, || {
            assert!(spawn_if_enabled().is_none());
        });
    }

    #[test]
    fn disabled_when_env_not_exactly_one() {
        for value in ["0", "true", "", "11"] {
            with_perf_log(Some(value), || {
                assert!(spawn_if_enabled().is_none(), "value {value:?} must disable");
            });
        }
    }

    #[test]
    fn unrecognized_value_is_reported_with_the_accepted_value() {
        for value in ["0", "true", "", "11"] {
            let msg = check_perf_log(Some(value)).unwrap_err();
            assert!(
                msg.contains("ARBVIS_PERF_LOG") && msg.contains("only `1` enables"),
                "message must name the var and the accepted value, got {msg:?}"
            );
        }
    }

    #[test]
    fn accepted_value_does_not_report() {
        assert_eq!(check_perf_log(Some("1")), Ok(true));
        assert_eq!(check_perf_log(None), Ok(false));
    }

    #[tokio::test]
    async fn enabled_spawns_task_and_stop_flag_ends_it() {
        with_perf_log(Some("1"), || {
            let stop = spawn_if_enabled().expect("ARBVIS_PERF_LOG=1 must spawn");
            assert!(!stop.load(Ordering::Relaxed));
            stop.store(true, Ordering::Relaxed);
        });
        // Give the spawned task a moment to observe the stop flag and exit;
        // the test only checks the handle contract, not the loop internals.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
