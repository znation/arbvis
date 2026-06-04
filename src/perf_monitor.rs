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

/// Spawn the monitor if `ARBVIS_PERF_LOG=1`. Returns a shutdown handle the
/// caller can drop to stop the task on exit.
pub fn spawn_if_enabled() -> Option<Arc<AtomicBool>> {
    if std::env::var("ARBVIS_PERF_LOG").ok().as_deref() != Some("1") {
        return None;
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
