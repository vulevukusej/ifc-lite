// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The empty-cut switch and its diagnostic counters.

use super::CutVerdict;
use std::sync::atomic::{AtomicU64, Ordering};

/// What the empty-cut test decided, summed over every pass since the last
/// [`take_empty_cut_stats`]. Diagnostics only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmptyCutStats {
    /// Openings judged.
    pub judged: u64,
    /// Openings left out: they remove nothing.
    pub skipped: u64,
    pub cuts_material: u64,
    pub inside_host: u64,
    pub clear_of_host: u64,
    /// Openings kept because the test did not apply (including every opening
    /// of a host that is not a set of closed solids).
    pub undecided: u64,
}

static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Count one verdict (`None`: the host could not be read, the opening kept).
pub fn record_empty_cut_verdict(verdict: Option<CutVerdict>) {
    let slot = match verdict {
        Some(CutVerdict::RemovesNothing) => 1,
        Some(CutVerdict::CutsMaterial) => 2,
        Some(CutVerdict::InsideHost) => 3,
        Some(CutVerdict::ClearOfHost) => 4,
        Some(CutVerdict::Undecided) | None => 5,
    };
    COUNTS[0].fetch_add(1, Ordering::Relaxed);
    COUNTS[slot].fetch_add(1, Ordering::Relaxed);
}

/// Drain the process-wide empty-cut counters.
pub fn take_empty_cut_stats() -> EmptyCutStats {
    let take = |i: usize| COUNTS[i].swap(0, Ordering::Relaxed);
    EmptyCutStats {
        judged: take(0),
        skipped: take(1),
        cuts_material: take(2),
        inside_host: take(3),
        clear_of_host: take(4),
        undecided: take(5),
    }
}

/// Whether the empty-cut skip runs: on unless `IFC_LITE_SKIP_EMPTY_CUTS=off`
/// (or `0`/`false`), read once per process.
pub fn skip_empty_cuts_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("IFC_LITE_SKIP_EMPTY_CUTS").unwrap_or_default();
        !matches!(value.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false")
    })
}
