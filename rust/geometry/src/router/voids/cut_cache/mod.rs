// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The cut cache: one boolean cut per group of identical voided hosts.
//!
//! Steel detailers (Tekla) export tens of thousands of hosts whose bodies and
//! openings repeat exactly, each copy placed somewhere else. The void path
//! cuts every copy afresh, although the cut of one copy, moved rigidly, is the
//! cut of the others. A pre-pass ([`CutCache::build`]) groups the voided hosts
//! by a structural key (`key.rs`): the host's type and body, and each opening's
//! type, body and placement relative to the host. Only hosts whose every
//! opening is placed directly relative to the host's own placement get a key,
//! so the openings sit in the host's frame identically in every copy.
//!
//! Each group's representative is its lowest express id. Its cut is computed
//! once, on whichever worker needs it first, through the ordinary uncached
//! path on the representative's own entity, so the representative's output is
//! exactly today's. Every other member gets that result moved by
//! `M = P_E · P_R⁻¹`, the element's resolved placement times the inverse of the
//! representative's (`map.rs`), when `M` is rigid; otherwise it is cut afresh.
//! A failed or partial cut is reused like any other: the representative's
//! output, whatever it was, is the copy's.
//!
//! The waiting rule keeps the output independent of scheduling: a worker that
//! finds the representative's cut in progress waits for it, or, if that worker
//! is itself the one computing it (a rayon steal re-entered the cache) or the
//! wait times out, computes the representative's cut privately. Either way a
//! copy receives the same representative result.
//!
//! What a copy no longer records: the CSG census, kernel failure logs, opening
//! classification counts, per-host opening diagnostics and rect-fast counters
//! of its own cut. The representative's are recorded once, by the job that
//! computed it, attributed to the representative's express id.
//! `IFC_LITE_CUT_CACHE=off` turns the cache off (read once per process).

mod key;
mod map;
mod stats;
#[cfg(test)]
mod tests;

use crate::router::GeometryRouter;
use crate::{Mesh, Result, SubMeshCollection};
use ifc_lite_core::{DecodedEntity, EntityDecoder};
use map::RigidMap;
use stats::{count, COUNTERS};
pub use stats::{
    cut_cache_enabled, cut_cache_set_enabled_override, take_cut_cache_groups, take_cut_cache_stats, CutCacheStats,
};
use nalgebra::Matrix4;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

/// How long a worker waits for another worker's representative cut before it
/// computes the cut itself. Long enough that a legitimate heavy cut finishes
/// first; finite so a rayon steal cycle between two waiting workers resolves.
const WAIT: Duration = Duration::from_secs(5);

/// The cut cache of one geometry pass, shared by every per-element router.
pub type SharedCutCache = Arc<CutCache>;

/// Groups of identical voided hosts and the representative cut of each.
pub struct CutCache {
    slots: Vec<Slot>,
    host_slot: FxHashMap<u32, u32>,
}

/// One group: its representative, and the representative's cut through each
/// of the two voided entries (per-item sub-meshes, merged frame parts).
pub(crate) struct Slot {
    rep: u32,
    styled: Cell<SubMeshCollection>,
    parts: Cell<Vec<Mesh>>,
}

/// A representative's finished cut and what moving it to a copy needs.
struct Computed<T> {
    output: T,
    /// The representative's resolved, scaled placement (`P_R`).
    placement: Matrix4<f64>,
    /// The representative's `IfcProductDefinitionShape`, for pairing its item
    /// ids with a copy's.
    shape: Option<u32>,
    /// The cut emptied the host: copies are consumed too.
    consumed: bool,
}

enum State<T> {
    Empty,
    Busy(ThreadId),
    Ready(Arc<Computed<T>>),
    Failed,
}

pub(crate) struct Cell<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
}

enum Acquire<'c, T> {
    Ready(Arc<Computed<T>>),
    Failed,
    Claimed(Claim<'c, T>),
    /// The cut is in progress on this thread further down the stack, or the
    /// wait timed out: compute it without publishing.
    Private,
}

/// The right to compute a cell. Dropping it unpublished (a decode failure or
/// an unwind) marks the cell failed, so no waiter waits on it.
struct Claim<'c, T> {
    cell: &'c Cell<T>,
    done: bool,
}

impl<T> Cell<T> {
    fn new() -> Self {
        Self { state: Mutex::new(State::Empty), ready: Condvar::new() }
    }

    fn acquire(&self) -> Acquire<'_, T> {
        let me = std::thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut timed_out = false;
        loop {
            match &*state {
                State::Ready(computed) => return Acquire::Ready(Arc::clone(computed)),
                State::Failed => return Acquire::Failed,
                State::Empty => {
                    *state = State::Busy(me);
                    return Acquire::Claimed(Claim { cell: self, done: false });
                }
                State::Busy(owner) if *owner == me || timed_out => return Acquire::Private,
                State::Busy(_) => {
                    let (next, wait) = self.ready.wait_timeout(state, WAIT).unwrap_or_else(|e| e.into_inner());
                    state = next;
                    timed_out = wait.timed_out();
                }
            }
        }
    }
}

impl<T> Claim<'_, T> {
    fn publish(mut self, computed: Option<Arc<Computed<T>>>) {
        self.finish(computed);
    }

    fn finish(&mut self, computed: Option<Arc<Computed<T>>>) {
        self.done = true;
        let mut state = self.cell.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = computed.map_or(State::Failed, State::Ready);
        self.cell.ready.notify_all();
    }
}

impl<T> Drop for Claim<'_, T> {
    fn drop(&mut self) {
        if !self.done {
            self.finish(None);
        }
    }
}

impl CutCache {
    fn slot(&self, host: u32) -> Option<&Slot> {
        self.host_slot.get(&host).and_then(|&index| self.slots.get(index as usize))
    }

    /// The representative of `host`'s group, when it has one.
    pub fn representative(&self, host: u32) -> Option<u32> {
        self.slot(host).map(|slot| slot.rep)
    }

    /// Number of groups (two or more identical hosts).
    pub fn group_count(&self) -> usize {
        self.slots.len()
    }
}

/// A voided entry's output, as the cache stores and moves it.
pub(crate) trait CutOutput: Clone + Send + Sync + Sized {
    fn cell(slot: &Slot) -> &Cell<Self>;
    fn compute(router: &GeometryRouter, host: &DecodedEntity, decoder: &mut EntityDecoder,
        void_index: &FxHashMap<u32, Vec<u32>>) -> Result<Self>;
    /// The item ids that must be paired with a copy's (none for frame parts).
    fn item_ids(&self) -> Vec<u32>;
    fn moved(&self, map: &RigidMap, ids: &FxHashMap<u32, u32>) -> Option<Self>;
}

impl CutOutput for SubMeshCollection {
    fn cell(slot: &Slot) -> &Cell<Self> {
        &slot.styled
    }

    fn compute(router: &GeometryRouter, host: &DecodedEntity, decoder: &mut EntityDecoder,
        void_index: &FxHashMap<u32, Vec<u32>>) -> Result<Self> {
        router.process_element_with_submeshes_and_voids_uncached(host, decoder, void_index)
    }

    fn item_ids(&self) -> Vec<u32> {
        self.sub_meshes.iter().map(|sub| sub.geometry_id).collect()
    }

    fn moved(&self, map: &RigidMap, ids: &FxHashMap<u32, u32>) -> Option<Self> {
        map::move_sub_meshes(self, map, ids)
    }
}

impl CutOutput for Vec<Mesh> {
    fn cell(slot: &Slot) -> &Cell<Self> {
        &slot.parts
    }

    fn compute(router: &GeometryRouter, host: &DecodedEntity, decoder: &mut EntityDecoder,
        void_index: &FxHashMap<u32, Vec<u32>>) -> Result<Self> {
        router.process_element_with_voids_parts_uncached(host, decoder, void_index)
    }

    fn item_ids(&self) -> Vec<u32> {
        Vec::new()
    }

    fn moved(&self, map: &RigidMap, _ids: &FxHashMap<u32, u32>) -> Option<Self> {
        self.iter().map(|mesh| map::move_mesh(mesh, map)).collect()
    }
}

impl GeometryRouter {
    /// Share `cache` with this router: its voided entries then reuse the
    /// representative cut of each group of identical hosts.
    pub fn enable_cut_cache(&mut self, cache: SharedCutCache) {
        self.execution_owner.changed();
        self.cut_cache = Some(cache);
    }

    /// `element`'s voided output through the cut cache, or `None` to take the
    /// uncached path (no cache, not grouped, or the move was refused).
    pub(crate) fn cached_cut<T: CutOutput>(&self, element: &DecodedEntity, decoder: &mut EntityDecoder,
        void_index: &FxHashMap<u32, Vec<u32>>) -> Option<Result<T>> {
        let slot = self.cut_cache.as_ref()?.slot(element.id)?;
        // Index-only hygiene is set per element from outside the key.
        if self.preserve_triangle_order.get() {
            return None;
        }
        let is_rep = slot.rep == element.id;
        match T::cell(slot).acquire() {
            Acquire::Failed => {
                count(&COUNTERS.failed);
                None
            }
            Acquire::Ready(computed) if is_rep => {
                count(&COUNTERS.representative_served);
                if computed.consumed {
                    self.record_void_consumed_host(element.id);
                }
                Some(Ok(computed.output.clone()))
            }
            Acquire::Ready(computed) => self.copy_of(&computed, element, decoder).map(Ok),
            Acquire::Claimed(claim) if is_rep => {
                let result = T::compute(self, element, decoder, void_index);
                claim.publish(self.capture(&result, element, decoder));
                count(&COUNTERS.computed);
                Some(result)
            }
            Acquire::Claimed(claim) => {
                let rep = decoder.decode_by_id(slot.rep).ok()?;
                let result = T::compute(self, &rep, decoder, void_index);
                let computed = self.capture(&result, &rep, decoder);
                claim.publish(computed.clone());
                count(&COUNTERS.computed);
                self.copy_of(&*computed?, element, decoder).map(Ok)
            }
            Acquire::Private if is_rep => None,
            Acquire::Private => {
                let rep = decoder.decode_by_id(slot.rep).ok()?;
                let result = T::compute(self, &rep, decoder, void_index);
                count(&COUNTERS.computed_privately);
                let computed = self.capture(&result, &rep, decoder)?;
                self.copy_of(&computed, element, decoder).map(Ok)
            }
        }
    }

    /// What a copy needs of the representative `rep`'s finished cut; `None`
    /// for an error result or an unplaced representative.
    fn capture<T: CutOutput>(&self, result: &Result<T>, rep: &DecodedEntity, decoder: &mut EntityDecoder)
        -> Option<Arc<Computed<T>>> {
        let output = result.as_ref().ok()?.clone();
        let placement = self.resolve_object_placement(rep, decoder).ok().filter(|p| p.is_placed())?;
        Some(Arc::new(Computed {
            output,
            placement: placement.transform,
            shape: crate::representation::structural::product_shape_id(rep),
            consumed: self.host_consumed_by_void(rep.id),
        }))
    }

    /// The representative's cut moved onto `element`, or `None` (counted as
    /// declined) when the move is refused.
    fn copy_of<T: CutOutput>(&self, computed: &Computed<T>, element: &DecodedEntity, decoder: &mut EntityDecoder)
        -> Option<T> {
        let moved = self.resolve_object_placement(element, decoder)
            .ok()
            .filter(|p| p.is_placed())
            .and_then(|placement| RigidMap::between(&computed.placement, &placement.transform, self.rtc_offset()))
            .and_then(|rigid| {
                let wanted = computed.output.item_ids();
                let ids = if wanted.is_empty() {
                    FxHashMap::default()
                } else {
                    let element_shape = crate::representation::structural::product_shape_id(element)?;
                    map::pair_ids(decoder, computed.shape?, element_shape, &wanted)?
                };
                computed.output.moved(&rigid, &ids)
            });
        match moved {
            Some(output) => {
                count(&COUNTERS.served);
                if computed.consumed {
                    self.record_void_consumed_host(element.id);
                }
                Some(output)
            }
            None => {
                count(&COUNTERS.declined);
                None
            }
        }
    }
}
