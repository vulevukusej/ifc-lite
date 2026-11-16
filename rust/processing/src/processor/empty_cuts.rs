// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Leave out the openings that remove nothing from their host
//! (`ifc_lite_geometry::empty_cut`) before the geometry pass cuts anything.
//! A host left with no opening is meshed uncut, as its body is drawn.
//! `IFC_LITE_SKIP_EMPTY_CUTS=off` keeps every opening.

use super::entity_index::ProcessingIndex;
use ifc_lite_geometry::empty_cut::{record_empty_cut_verdict, skip_empty_cuts_enabled, CutVerdict, HostBody};
use ifc_lite_geometry::{GeometryRouter, RouterConfig};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Hosts judged per parallel task, each task with its own decoder and router.
const CHUNK: usize = 64;

/// `void_index` without the openings that remove nothing. `config` is the
/// pass's router configuration; `plane_angle` seeds each task's decoder.
pub(super) fn drop_empty_cuts(
    content: &[u8],
    entity_index: &ProcessingIndex,
    void_index: FxHashMap<u32, Vec<u32>>,
    config: &RouterConfig,
    plane_angle: f64,
) -> FxHashMap<u32, Vec<u32>> {
    if !skip_empty_cuts_enabled() || void_index.is_empty() {
        return void_index;
    }
    let mut hosts: Vec<(u32, Vec<u32>)> = void_index.into_iter().collect();
    hosts.sort_unstable_by_key(|(id, _)| *id);
    hosts.par_chunks_mut(CHUNK).for_each(|chunk| {
        let mut decoder = entity_index.decoder(content).with_strict_reals();
        decoder.seed_unit_scales(config.unit_scale, plane_angle);
        let router = GeometryRouter::from_config(config.clone());
        for (host, openings) in chunk.iter_mut() {
            let body = decoder.decode_by_id(*host).ok().and_then(|host| HostBody::of(&router, &host, &mut decoder));
            openings.retain(|&opening| {
                let verdict = body.as_ref().map(|body| body.judge(&router, opening, &mut decoder));
                record_empty_cut_verdict(verdict);
                verdict != Some(CutVerdict::RemovesNothing)
            });
        }
    });
    hosts.into_iter().filter(|(_, openings)| !openings.is_empty()).collect()
}
