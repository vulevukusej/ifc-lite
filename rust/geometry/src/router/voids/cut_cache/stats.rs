// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The cut cache's switch and its diagnostic counters.

use std::sync::atomic::{AtomicI8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// What the cache did, summed over every pass since the last
/// [`take_cut_cache_stats`]. Diagnostics only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CutCacheStats {
    /// Groups of two or more identical hosts.
    pub groups: u64,
    /// Hosts in those groups.
    pub grouped_hosts: u64,
    /// Voided hosts the pre-pass could not key (an opening placed other than
    /// directly relative to the host, a missing body), cut afresh.
    pub unkeyed_hosts: u64,
    /// Representative cuts computed and published.
    pub computed: u64,
    /// Representative cuts computed privately (re-entry or wait timeout).
    pub computed_privately: u64,
    /// Copies served by moving the representative's cut.
    pub served: u64,
    /// Representatives served their own cut computed by another job.
    pub representative_served: u64,
    /// Copies cut afresh because the move was refused (not rigid, unpaired
    /// item ids, textures, far from the RTC origin).
    pub declined: u64,
    /// Lookups that found the representative's cut failed (an error result).
    pub failed: u64,
}

pub(super) struct Counters {
    pub(super) groups: AtomicU64,
    pub(super) grouped_hosts: AtomicU64,
    pub(super) unkeyed_hosts: AtomicU64,
    pub(super) computed: AtomicU64,
    pub(super) computed_privately: AtomicU64,
    pub(super) served: AtomicU64,
    pub(super) representative_served: AtomicU64,
    pub(super) declined: AtomicU64,
    pub(super) failed: AtomicU64,
}

pub(super) static COUNTERS: Counters = Counters {
    groups: AtomicU64::new(0),
    grouped_hosts: AtomicU64::new(0),
    unkeyed_hosts: AtomicU64::new(0),
    computed: AtomicU64::new(0),
    computed_privately: AtomicU64::new(0),
    served: AtomicU64::new(0),
    representative_served: AtomicU64::new(0),
    declined: AtomicU64::new(0),
    failed: AtomicU64::new(0),
};

pub(super) fn count(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// The groups built since the last [`take_cut_cache_groups`], each in express
/// id order (its representative first), for checking the cache's output
/// against the uncached path. Diagnostics only.
pub(super) static GROUPS: Mutex<Vec<Vec<u32>>> = Mutex::new(Vec::new());

/// Drain the groups of every cache built since the last call.
pub fn take_cut_cache_groups() -> Vec<Vec<u32>> {
    std::mem::take(&mut *GROUPS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Drain the process-wide cut cache counters.
pub fn take_cut_cache_stats() -> CutCacheStats {
    let take = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed);
    let c = &COUNTERS;
    CutCacheStats {
        groups: take(&c.groups),
        grouped_hosts: take(&c.grouped_hosts),
        unkeyed_hosts: take(&c.unkeyed_hosts),
        computed: take(&c.computed),
        computed_privately: take(&c.computed_privately),
        served: take(&c.served),
        representative_served: take(&c.representative_served),
        declined: take(&c.declined),
        failed: take(&c.failed),
    }
}

static ENABLED_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

/// Test/harness hook: force the cache on or off, or `None` for the
/// environment default. Process-global, like
/// [`crate::local_frame_set_enabled_override`].
pub fn cut_cache_set_enabled_override(enabled: Option<bool>) {
    ENABLED_OVERRIDE.store(enabled.map_or(-1, i8::from), Ordering::Relaxed);
}

/// Whether [`CutCache::build`] builds a cache: on unless
/// `IFC_LITE_CUT_CACHE=off` (or `0`/`false`), read once per process.
pub fn cut_cache_enabled() -> bool {
    match ENABLED_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("IFC_LITE_CUT_CACHE").unwrap_or_default();
        !matches!(value.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false")
    })
}
