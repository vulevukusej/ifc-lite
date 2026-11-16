// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The cut cache's pre-pass: a structural key per voided host, and the groups
//! of hosts that share one.
//!
//! The key is everything the cut reads that a copy could change, by value:
//! the host's IFC type and its whole `IfcProductDefinitionShape`, and for each
//! opening its IFC type, its `PredefinedType`, its whole shape and its
//! `RelativePlacement`. The openings are a sorted multiset, so two copies whose
//! voids relations were written in a different order still share a key. A host
//! has no key when any opening is placed other than by an `IfcLocalPlacement`
//! whose `PlacementRelTo` is the host's own `ObjectPlacement`: only then is the
//! opening's frame in the host's frame the `RelativePlacement` alone.
//!
//! Structural hashing is `content_hash::item_signature` (renumbering-invariant,
//! references followed), the same hash the item dedup cache keys on.

use super::stats::{COUNTERS, GROUPS};
use super::{Cell, CutCache, Slot};
use crate::router::content_hash::{fold, fold_bytes, item_signature};
use crate::representation::structural::product_shape_id;
use ifc_lite_core::{slot, EntityDecoder, IfcType};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const KEY_TAG: u64 = 0xC07C_AC4E_0000_0001;
const OPENING_TAG: u64 = 0xC07C_AC4E_0000_0002;
/// Hosts keyed per parallel task, each task with its own decoder and memo.
const CHUNK: usize = 256;
/// `IfcLocalPlacement.PlacementRelTo` and `.RelativePlacement`.
const PLACEMENT_REL_TO: usize = 0;
const RELATIVE_PLACEMENT: usize = 1;
/// `PredefinedType` of `IfcOpeningElement` (IFC4) and `IfcVoidingFeature`.
const OPENING_PREDEFINED_TYPE: usize = 8;

/// Per-task hashing state: one memo per decoder, since it keys on entity ids.
pub(super) struct Hasher {
    memo: FxHashMap<u32, u128>,
    refused: usize,
}

impl Hasher {
    pub(super) fn new() -> Self {
        Self { memo: FxHashMap::default(), refused: 0 }
    }

    fn signature(&mut self, decoder: &mut EntityDecoder, id: u32) -> u128 {
        item_signature(decoder, id, &mut self.memo, &mut self.refused)
    }
}

fn fold_u128(state: u128, value: u128) -> u128 {
    fold(fold(state, value as u64), (value >> 64) as u64)
}

/// The key of host `host_id` voided by `openings`, or `None` when the host
/// cannot share its cut (see the module doc).
pub(super) fn host_key(decoder: &mut EntityDecoder, hasher: &mut Hasher, host_id: u32, openings: &[u32])
    -> Option<u128> {
    let host = decoder.decode_by_id(host_id).ok()?;
    let host_placement = host.get_ref(slot::IfcProduct::OBJECT_PLACEMENT.index())?;
    let shape = product_shape_id(&host)?;
    let mut sigs = Vec::with_capacity(openings.len());
    for &opening in openings {
        sigs.push(opening_signature(decoder, hasher, opening, host_placement)?);
    }
    sigs.sort_unstable();
    let mut key = fold_bytes(fold(0, KEY_TAG), host.ifc_type.to_string().as_bytes());
    key = fold_u128(key, hasher.signature(decoder, shape));
    key = fold(key, sigs.len() as u64);
    Some(sigs.into_iter().fold(key, fold_u128))
}

fn opening_signature(decoder: &mut EntityDecoder, hasher: &mut Hasher, id: u32, host_placement: u32)
    -> Option<u128> {
    let opening = decoder.decode_by_id(id).ok()?;
    let placement = decoder.decode_by_id(opening.get_ref(slot::IfcProduct::OBJECT_PLACEMENT.index())?).ok()?;
    if placement.ifc_type != IfcType::IfcLocalPlacement || placement.get_ref(PLACEMENT_REL_TO)? != host_placement {
        return None;
    }
    let relative = placement.get_ref(RELATIVE_PLACEMENT)?;
    let shape = product_shape_id(&opening)?;
    let mut sig = fold_bytes(fold(0, OPENING_TAG), opening.ifc_type.to_string().as_bytes());
    sig = fold_bytes(sig, format!("{:?}", opening.get(OPENING_PREDEFINED_TYPE)).as_bytes());
    sig = fold_u128(sig, hasher.signature(decoder, shape));
    Some(fold_u128(sig, hasher.signature(decoder, relative)))
}

impl CutCache {
    /// Group the hosts of `void_index` by key. `decoder` makes one decoder
    /// over the pass's source per parallel task; `excluded` names hosts that
    /// must keep the uncached path (a material-layer host, whose sub-mesh ids
    /// are material ids chosen per element). `None` when the cache is off
    /// ([`super::cut_cache_enabled`]) or no two hosts share a key.
    pub fn build<'a, D, X>(void_index: &FxHashMap<u32, Vec<u32>>, decoder: D, excluded: X) -> Option<Arc<Self>>
    where
        D: Fn() -> EntityDecoder<'a> + Sync,
        X: Fn(u32) -> bool + Sync,
    {
        if !super::cut_cache_enabled() {
            return None;
        }
        let mut hosts: Vec<(u32, &[u32])> = void_index
            .iter()
            .filter(|(&id, openings)| !openings.is_empty() && !excluded(id))
            .map(|(&id, openings)| (id, openings.as_slice()))
            .collect();
        hosts.sort_unstable_by_key(|&(id, _)| id);
        let keys: Vec<(u32, Option<u128>)> = hosts
            .par_chunks(CHUNK)
            .flat_map_iter(|chunk| {
                let mut decoder = decoder();
                let mut hasher = Hasher::new();
                chunk
                    .iter()
                    .map(|&(id, openings)| (id, host_key(&mut decoder, &mut hasher, id, openings)))
                    .collect::<Vec<_>>()
            })
            .collect();
        Some(Arc::new(Self::from_keys(&keys))).filter(|cache| !cache.slots.is_empty())
    }

    /// Groups from `(host, key)` pairs sorted by host id: the first member of
    /// each group of two or more is its representative.
    pub(super) fn from_keys(keys: &[(u32, Option<u128>)]) -> Self {
        let mut members: FxHashMap<u128, Vec<u32>> = FxHashMap::default();
        let mut unkeyed = 0u64;
        for &(id, key) in keys {
            match key {
                Some(key) => members.entry(key).or_default().push(id),
                None => unkeyed += 1,
            }
        }
        let mut groups: Vec<Vec<u32>> = members.into_values().filter(|ids| ids.len() > 1).collect();
        groups.sort_unstable_by_key(|ids| ids[0]);
        let mut cache = Self { slots: Vec::with_capacity(groups.len()), host_slot: FxHashMap::default() };
        GROUPS.lock().unwrap_or_else(|e| e.into_inner()).extend(groups.iter().cloned());
        for ids in groups {
            let index = cache.slots.len() as u32;
            cache.host_slot.extend(ids.iter().map(|&id| (id, index)));
            cache.slots.push(Slot { rep: ids[0], styled: Cell::new(), parts: Cell::new() });
        }
        COUNTERS.groups.fetch_add(cache.slots.len() as u64, Ordering::Relaxed);
        COUNTERS.grouped_hosts.fetch_add(cache.host_slot.len() as u64, Ordering::Relaxed);
        COUNTERS.unkeyed_hosts.fetch_add(unkeyed, Ordering::Relaxed);
        cache
    }
}
