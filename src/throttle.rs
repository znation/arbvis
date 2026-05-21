//! Adaptive (AIMD) concurrency throttle for outbound HTTP, tokio-native.
//!
//! All Hub-bound HTTP in this crate goes through [`Throttle::global`]. Tasks
//! `acquire().await` a permit before sending a request and report the outcome:
//!
//! - [`Throttle::record_success`] — gradually scale up (+1 worker every ≥10s
//!   after 50 successes, gated by 60s cooldowns since last 429/timeout).
//! - [`Throttle::record_rate_limit`] — halve the active limit immediately (floor
//!   `max(4, max/64)`) and reset cooldown timers.
//! - [`Throttle::record_timeout`] — reduce by 10% (floor `max(4, max/16)`) and
//!   reset cooldown timers.
//!
//! Use [`with_throttle`] to wrap an async call: acquire, run, classify the
//! error, retry with decorrelated jitter where appropriate, and record the
//! outcome.
//!
//! `max_workers` defaults to [`MAX_FETCH_WORKERS`] (128) — well above num_cpus —
//! so the network parallelism is decoupled from CPU parallelism. Fetch
//! workers above `active_limit` park on `scale_up_notify`.
//!
//! The math is a direct port of `xetcas/sizzle_sync/src/commands/subscriber.rs`.

use std::future::Future;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use tokio::sync::Notify;

/// Workers the throttle starts at (matches sizzle_sync).
const INITIAL_WORKERS: usize = 4;
/// Default ceiling on concurrent in-flight HTTP requests. Decoupled from
/// num_cpus because the throttle's job is network parallelism, not CPU.
pub const MAX_FETCH_WORKERS: usize = 128;
/// Successful fetches required before each scale-up tick. Acts as a sanity
/// check: if requests aren't actually completing, don't scale up.
const SUCCESSES_TO_SCALE_UP: usize = 10;
/// Minimum interval between scale-ups. Combined with multiplicative
/// slow-start (`×2` per tick) and 5 doublings from 4 → 128, this gives a
/// total scale-up time of roughly `5 × SCALE_UP_INTERVAL_SECS` seconds.
/// 90 s yields ~7.5 min to reach max — fast enough to recover from a long
/// run, slow enough to give the upstream room to push back via 429s if it
/// doesn't like the load.
const SCALE_UP_INTERVAL_SECS: i64 = 90;
/// Cooldown after a rate limit before scaling up resumes.
const RATE_LIMIT_COOLDOWN_SECS: i64 = 60;
/// Cooldown after a timeout before scaling up resumes.
const TIMEOUT_COOLDOWN_SECS: i64 = 60;
/// Maximum 429 retries for a single operation before giving up.
const MAX_RATE_LIMIT_RETRIES: u32 = 10;
/// Maximum transient/timeout retries for a single operation before giving up.
const MAX_TIMEOUT_RETRIES: u32 = 5;

/// Decorrelated jitter base (sizzle_sync uses 1s).
const BACKOFF_BASE_MS: u64 = 1000;
/// Decorrelated jitter cap (sizzle_sync uses 64s).
const BACKOFF_CAP_MS: u64 = 64_000;

/// Per-process AIMD throttle. Use [`Throttle::global`] in production; tests in
/// this module instantiate fresh ones with [`Throttle::new_for_test`].
pub struct Throttle {
    max_workers: usize,
    active_limit: AtomicUsize,
    in_flight: AtomicUsize,
    last_rate_limit: AtomicI64,
    last_timeout: AtomicI64,
    last_scale_up: AtomicI64,
    successes_since_scale_up: AtomicUsize,
    /// Monotone counter feeding the decorrelated-jitter RNG so different
    /// tasks/operations land on different sleeps.
    jitter_nonce: AtomicU64,
    /// Notified when a permit is released or when `active_limit` rises. Parked
    /// acquirers re-check `in_flight < active_limit` after a wake.
    scale_up_notify: Notify,
    // Cumulative perf counters surfaced via `stats()` for the perf monitor.
    total_rate_limits: AtomicU64,
    total_timeouts: AtomicU64,
    /// Tasks currently sleeping in a `with_throttle` backoff between retries.
    /// A stall during which `in_flight == 0` but `in_backoff > 0` is an AIMD
    /// backoff stall, not idle.
    in_backoff: AtomicUsize,
}

#[derive(Clone, Copy, Debug)]
pub struct ThrottleStats {
    pub in_flight: usize,
    pub active_limit: usize,
    pub max_workers: usize,
    pub in_backoff: usize,
    pub total_rate_limits: u64,
    pub total_timeouts: u64,
}

impl Throttle {
    fn new(max_workers: usize) -> Self {
        let initial = INITIAL_WORKERS.min(max_workers.max(1));
        Self {
            max_workers: max_workers.max(1),
            active_limit: AtomicUsize::new(initial),
            in_flight: AtomicUsize::new(0),
            last_rate_limit: AtomicI64::new(0),
            last_timeout: AtomicI64::new(0),
            last_scale_up: AtomicI64::new(0),
            successes_since_scale_up: AtomicUsize::new(0),
            jitter_nonce: AtomicU64::new(0),
            scale_up_notify: Notify::new(),
            total_rate_limits: AtomicU64::new(0),
            total_timeouts: AtomicU64::new(0),
            in_backoff: AtomicUsize::new(0),
        }
    }

    /// Snapshot of perf counters. Lock-free; safe to call at any time.
    pub fn stats(&self) -> ThrottleStats {
        ThrottleStats {
            in_flight: self.in_flight.load(Ordering::Relaxed),
            active_limit: self.active_limit.load(Ordering::Relaxed),
            max_workers: self.max_workers,
            in_backoff: self.in_backoff.load(Ordering::Relaxed),
            total_rate_limits: self.total_rate_limits.load(Ordering::Relaxed),
            total_timeouts: self.total_timeouts.load(Ordering::Relaxed),
        }
    }

    /// Global throttle, capped at [`MAX_FETCH_WORKERS`]. Tokio runtime must be
    /// active when `acquire().await` is called, but this constructor itself
    /// does no runtime-requiring work.
    pub fn global() -> &'static Self {
        static G: OnceLock<Throttle> = OnceLock::new();
        G.get_or_init(|| Throttle::new(MAX_FETCH_WORKERS))
    }

    #[cfg(test)]
    fn new_for_test(max_workers: usize) -> Self {
        Self::new(max_workers)
    }

    /// Await a permit. Returns when `in_flight` becomes less than
    /// `active_limit`. The permit decrements `in_flight` on drop and wakes one
    /// waiter via `scale_up_notify`.
    pub async fn acquire(&self) -> Permit<'_> {
        loop {
            // Register interest before reading state so a notify between the
            // check and the await isn't lost.
            let waiter = self.scale_up_notify.notified();
            tokio::pin!(waiter);
            waiter.as_mut().enable();

            let limit = self.active_limit.load(Ordering::SeqCst);
            let cur = self.in_flight.load(Ordering::SeqCst);
            if cur < limit
                && self
                    .in_flight
                    .compare_exchange(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                return Permit { throttle: self };
            }
            waiter.as_mut().await;
        }
    }

    /// Notify all waiters. Used on scale-up and on permit drop.
    fn wake_waiters(&self) {
        // `notify_waiters` wakes everyone currently waiting (without leaving a
        // permit token behind), which is what we want — every waiter re-checks
        // its predicate under SeqCst loads.
        self.scale_up_notify.notify_waiters();
    }

    /// Record a successful HTTP exchange. May scale up if all gates pass.
    pub fn record_success(&self) {
        let current_limit = self.active_limit.load(Ordering::SeqCst);
        if current_limit >= self.max_workers {
            return;
        }

        let now = unix_now();
        let last_rate_limit = self.last_rate_limit.load(Ordering::SeqCst);
        let last_timeout = self.last_timeout.load(Ordering::SeqCst);
        let last_scale_up = self.last_scale_up.load(Ordering::SeqCst);

        if !(last_rate_limit == 0 || now - last_rate_limit > RATE_LIMIT_COOLDOWN_SECS) {
            return;
        }
        if !(last_timeout == 0 || now - last_timeout > TIMEOUT_COOLDOWN_SECS) {
            return;
        }
        if !(last_scale_up == 0 || now - last_scale_up >= SCALE_UP_INTERVAL_SECS) {
            return;
        }

        // CAS-on-counter: only the task that wins the `successes` → 0 swap
        // performs the scale-up.
        let successes = self.successes_since_scale_up.fetch_add(1, Ordering::SeqCst) + 1;
        if successes < SUCCESSES_TO_SCALE_UP {
            return;
        }
        if self
            .successes_since_scale_up
            .compare_exchange(successes, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }

        // TCP-slow-start: while we've never seen a 429 or timeout, *double*
        // active_limit each scale-up tick (capped at max_workers). After any
        // backoff event we switch to additive +1 — the multiplicative-decrease
        // half of AIMD then dominates and we conservatively probe upward.
        // Without slow-start, getting from 4 → 128 workers would take ~124
        // ticks; with it, only ~5.
        let in_slow_start = last_rate_limit == 0 && last_timeout == 0;
        let scaled =
            self.active_limit
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                    if current >= self.max_workers {
                        return None;
                    }
                    let next = if in_slow_start {
                        current.saturating_mul(2).min(self.max_workers)
                    } else {
                        current + 1
                    };
                    Some(next)
                });
        if let Ok(prev) = scaled {
            let new_limit = self.active_limit.load(Ordering::SeqCst);
            self.last_scale_up.store(now, Ordering::SeqCst);
            log::info!(
                "throttle: scaled up to {} workers (max {}, prev {}{})",
                new_limit,
                self.max_workers,
                prev,
                if in_slow_start { ", slow-start" } else { "" },
            );
            self.wake_waiters();
        }
    }

    /// Record a rate-limit (429). Halve the active limit with a tight floor.
    pub fn record_rate_limit(&self) {
        self.total_rate_limits.fetch_add(1, Ordering::Relaxed);
        let now = unix_now();
        self.last_rate_limit.store(now, Ordering::SeqCst);
        let floor = (self.max_workers / 64).max(4).min(self.max_workers);
        let prev = self.active_limit.load(Ordering::SeqCst);
        let new_limit =
            self.active_limit
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                    let new = (current / 2).max(floor);
                    if new != current {
                        Some(new)
                    } else {
                        None
                    }
                });
        self.successes_since_scale_up.store(0, Ordering::SeqCst);
        if new_limit.is_ok() {
            let now_limit = self.active_limit.load(Ordering::SeqCst);
            log::warn!(
                "throttle: rate limited; reducing concurrency {prev} → {now_limit} (floor {floor})",
            );
        }
    }

    /// Record a transient/timeout failure. Reduce by 10% with a gentler floor.
    pub fn record_timeout(&self) {
        self.total_timeouts.fetch_add(1, Ordering::Relaxed);
        let floor = (self.max_workers / 16).max(4).min(self.max_workers);
        let reduced =
            self.active_limit
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                    let new = ((current * 9) / 10).max(floor);
                    if new < current {
                        Some(new)
                    } else {
                        None
                    }
                });
        if reduced.is_ok() {
            self.successes_since_scale_up.store(0, Ordering::SeqCst);
            self.last_timeout.store(unix_now(), Ordering::SeqCst);
            let now_limit = self.active_limit.load(Ordering::SeqCst);
            log::warn!(
                "throttle: transient error; reducing concurrency to {now_limit} (floor {floor})",
            );
        }
    }

    /// Sleep duration for the `attempt`-th 429 retry (1-indexed).
    pub fn rate_limit_backoff(&self, attempt: u32) -> Duration {
        let base_ms = BACKOFF_BASE_MS;
        let cap_ms = BACKOFF_CAP_MS;
        let pow = (attempt.saturating_sub(1)).min(3);
        let prev_ms = base_ms.saturating_mul(4u64.pow(pow));
        let span = (prev_ms.saturating_mul(3)).saturating_sub(base_ms) + 1;
        let r = self.next_rand();
        let jitter_ms = base_ms + (r % span);
        Duration::from_millis(jitter_ms.min(cap_ms))
    }

    pub fn timeout_backoff(&self, _attempt: u32) -> Duration {
        Duration::from_secs(2)
    }

    fn next_rand(&self) -> u64 {
        let n = self.jitter_nonce.fetch_add(1, Ordering::Relaxed);
        splitmix64(n.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// Current AIMD-allowed concurrent HTTP request ceiling. Used by the UI
    /// layer to render a throttle status line.
    #[inline]
    pub fn active_limit(&self) -> usize {
        self.active_limit.load(Ordering::Relaxed)
    }

    /// Current in-flight HTTP requests (≤ [`Self::active_limit`] except briefly
    /// after a scale-down, when in-flight may exceed the new limit until
    /// outstanding permits drop).
    #[inline]
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Configured ceiling on the AIMD `active_limit`. Constant for the
    /// lifetime of the process — used as the `len` for a throttle progress bar.
    #[inline]
    pub fn max_workers(&self) -> usize {
        self.max_workers
    }
}

/// RAII permit. Decrements `in_flight` on drop and wakes parked acquirers.
pub struct Permit<'a> {
    throttle: &'a Throttle,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.throttle.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.throttle.wake_waiters();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    RateLimit,
    Timeout,
    Permanent,
}

pub trait ErrorClassify {
    fn classify(&self) -> Outcome;
}

impl ErrorClassify for hf_hub::HFError {
    fn classify(&self) -> Outcome {
        match self {
            hf_hub::HFError::RateLimited { .. } => Outcome::RateLimit,
            hf_hub::HFError::Request { source, .. } => {
                if source.is_connect() || source.is_timeout() || source.is_request() {
                    Outcome::Timeout
                } else {
                    Outcome::Permanent
                }
            }
            hf_hub::HFError::Http { context } => match context.status.as_u16() {
                429 => Outcome::RateLimit,
                500 | 502 | 503 | 504 => Outcome::Timeout,
                _ => Outcome::Permanent,
            },
            _ => Outcome::Permanent,
        }
    }
}

impl ErrorClassify for reqwest::Error {
    fn classify(&self) -> Outcome {
        if let Some(status) = self.status() {
            return match status.as_u16() {
                429 => Outcome::RateLimit,
                500 | 502 | 503 | 504 => Outcome::Timeout,
                _ => Outcome::Permanent,
            };
        }
        if self.is_timeout() || self.is_connect() || self.is_request() {
            Outcome::Timeout
        } else {
            Outcome::Permanent
        }
    }
}

/// Run an async `op` under a throttle permit with AIMD retries.
/// The permit is held only while the future is running, and dropped before
/// any sleep so other tasks can keep working.
pub async fn with_throttle<T, E, F, Fut>(label: &str, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: ErrorClassify,
{
    let throttle = Throttle::global();
    let mut rate_limit_retries: u32 = 0;
    let mut timeout_retries: u32 = 0;

    loop {
        let permit = throttle.acquire().await;
        let result = op().await;
        drop(permit);

        match result {
            Ok(v) => {
                throttle.record_success();
                return Ok(v);
            }
            Err(e) => match e.classify() {
                Outcome::RateLimit => {
                    rate_limit_retries += 1;
                    throttle.record_rate_limit();
                    if rate_limit_retries > MAX_RATE_LIMIT_RETRIES {
                        log::warn!(
                            "{label}: giving up after {rate_limit_retries} rate-limit retries",
                        );
                        return Err(e);
                    }
                    let delay = throttle.rate_limit_backoff(rate_limit_retries);
                    log::debug!(
                        "{label}: rate-limited; sleeping {:.1}s before retry {}/{}",
                        delay.as_secs_f32(),
                        rate_limit_retries,
                        MAX_RATE_LIMIT_RETRIES,
                    );
                    throttle.in_backoff.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(delay).await;
                    throttle.in_backoff.fetch_sub(1, Ordering::Relaxed);
                }
                Outcome::Timeout => {
                    timeout_retries += 1;
                    throttle.record_timeout();
                    if timeout_retries > MAX_TIMEOUT_RETRIES {
                        log::warn!("{label}: giving up after {timeout_retries} transient retries",);
                        return Err(e);
                    }
                    let delay = throttle.timeout_backoff(timeout_retries);
                    log::debug!(
                        "{label}: transient error; sleeping {:.1}s before retry {}/{}",
                        delay.as_secs_f32(),
                        timeout_retries,
                        MAX_TIMEOUT_RETRIES,
                    );
                    throttle.in_backoff.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(delay).await;
                    throttle.in_backoff.fetch_sub(1, Ordering::Relaxed);
                }
                Outcome::Permanent => return Err(e),
            },
        }
    }
}

fn unix_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_rate_limit_halves_with_floor() {
        let t = Throttle::new_for_test(640);
        t.active_limit.store(640, Ordering::SeqCst);
        t.record_rate_limit();
        assert_eq!(t.active_limit(), 320);
        for _ in 0..20 {
            t.record_rate_limit();
        }
        assert_eq!(t.active_limit(), 10);
    }

    #[test]
    fn record_rate_limit_small_max_floors_at_four() {
        let t = Throttle::new_for_test(8);
        t.active_limit.store(8, Ordering::SeqCst);
        for _ in 0..10 {
            t.record_rate_limit();
        }
        assert_eq!(t.active_limit(), 4);
    }

    #[test]
    fn record_timeout_reduces_by_ten_percent_with_floor() {
        let t = Throttle::new_for_test(160);
        t.active_limit.store(100, Ordering::SeqCst);
        t.record_timeout();
        assert_eq!(t.active_limit(), 90);
        for _ in 0..200 {
            t.record_timeout();
        }
        assert_eq!(t.active_limit(), 10);
    }

    #[test]
    fn record_timeout_at_floor_does_not_update_last_timeout() {
        let t = Throttle::new_for_test(160);
        t.active_limit.store(10, Ordering::SeqCst);
        t.last_timeout.store(0, Ordering::SeqCst);
        t.record_timeout();
        assert_eq!(t.last_timeout.load(Ordering::SeqCst), 0);
        assert_eq!(t.active_limit(), 10);
    }

    #[test]
    fn record_success_gated_by_scale_up_interval() {
        let t = Throttle::new_for_test(64);
        t.active_limit.store(10, Ordering::SeqCst);
        t.last_scale_up.store(unix_now(), Ordering::SeqCst);
        for _ in 0..200 {
            t.record_success();
        }
        assert_eq!(t.active_limit(), 10);
    }

    #[test]
    fn record_success_gated_by_rate_limit_cooldown() {
        let t = Throttle::new_for_test(64);
        t.active_limit.store(10, Ordering::SeqCst);
        t.last_rate_limit.store(unix_now(), Ordering::SeqCst);
        for _ in 0..200 {
            t.record_success();
        }
        assert_eq!(t.active_limit(), 10);
    }

    #[test]
    fn record_success_doubles_in_slow_start() {
        // Slow-start: no prior 429 or timeout → multiplicative growth (×2).
        let t = Throttle::new_for_test(64);
        t.active_limit.store(10, Ordering::SeqCst);
        for _ in 0..SUCCESSES_TO_SCALE_UP {
            t.record_success();
        }
        assert_eq!(t.active_limit(), 20);
        for _ in 0..(SUCCESSES_TO_SCALE_UP - 1) {
            t.record_success();
        }
        assert_eq!(
            t.active_limit(),
            20,
            "should not scale before the next success burst"
        );
    }

    #[test]
    fn record_success_adds_one_after_backoff() {
        // Once any 429/timeout has been seen, slow-start ends and growth is +1.
        let t = Throttle::new_for_test(64);
        t.active_limit.store(10, Ordering::SeqCst);
        // Mark a timeout in the distant past so the cooldown gate is already passed
        // but slow-start is permanently exited.
        t.last_timeout
            .store(unix_now() - TIMEOUT_COOLDOWN_SECS - 1, Ordering::SeqCst);
        for _ in 0..SUCCESSES_TO_SCALE_UP {
            t.record_success();
        }
        assert_eq!(t.active_limit(), 11);
    }

    #[test]
    fn slow_start_caps_at_max_workers() {
        let t = Throttle::new_for_test(128);
        t.active_limit.store(100, Ordering::SeqCst);
        for _ in 0..SUCCESSES_TO_SCALE_UP {
            t.record_success();
        }
        // 100 × 2 = 200, clamped to max (128).
        assert_eq!(t.active_limit(), 128);
    }

    #[test]
    fn record_success_no_op_at_max() {
        let t = Throttle::new_for_test(8);
        t.active_limit.store(8, Ordering::SeqCst);
        for _ in 0..1000 {
            t.record_success();
        }
        assert_eq!(t.active_limit(), 8);
    }

    #[test]
    fn rate_limit_backoff_within_bounds() {
        let t = Throttle::new_for_test(16);
        for attempt in 1u32..=10 {
            let d = t.rate_limit_backoff(attempt);
            assert!(d >= Duration::from_millis(BACKOFF_BASE_MS));
            assert!(d <= Duration::from_millis(BACKOFF_CAP_MS));
        }
    }

    #[tokio::test]
    async fn acquire_parks_when_at_limit_and_wakes_on_release() {
        use std::sync::Arc;
        use std::time::Instant;

        let t = Arc::new(Throttle::new_for_test(16));
        t.active_limit.store(1, Ordering::SeqCst);

        let p1 = t.acquire().await;
        assert_eq!(t.in_flight(), 1);

        let t2 = Arc::clone(&t);
        let start = Instant::now();
        let handle = tokio::spawn(async move {
            let _p2 = t2.acquire().await;
        });

        // Give the spawned task a chance to park.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(t.in_flight(), 1, "spawned task should not have acquired");

        drop(p1);
        handle.await.unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(t.in_flight(), 0);
    }

    #[tokio::test]
    async fn acquire_wakes_on_scale_up() {
        use std::sync::Arc;

        let t = Arc::new(Throttle::new_for_test(16));
        t.active_limit.store(1, Ordering::SeqCst);

        let _p1 = t.acquire().await;

        let t2 = Arc::clone(&t);
        let handle = tokio::spawn(async move {
            let _p2 = t2.acquire().await;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(t.in_flight(), 1);

        t.active_limit.store(2, Ordering::SeqCst);
        t.wake_waiters();

        handle.await.unwrap();
        assert_eq!(t.in_flight(), 1);
    }
}
