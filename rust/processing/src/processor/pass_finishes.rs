// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The native pass's `on_finishes` event (#8341 C4-13a), one entry per mesh
//! of the batch about to be emitted.

use super::ProcessingSink;
use crate::element::StyleSource;
use crate::style::ModelFinishes;
use crate::types::mesh::MeshData;

/// `on_finishes` for the batch about to be emitted, when the request asked
/// for finishes: each mesh's finish from the source its colour came from
/// (#10698), never from its colour.
///
/// `styles_deferred`: a `fast_first` pass meshes its first chunk before the
/// attached styles are resolved, and the replay after it sends colours only.
/// So every batch of that chunk was coloured without the item styles, and a
/// mesh's recorded source may be a material, the type default or a fallback
/// it would not take once they resolve. A mesh whose own representation item
/// reaches a styled item (itself, or through the mapped chase below it) takes
/// that item's finish instead, from the join's own fully resolved index: the
/// item-level step, which outranks every other once the styles are known.
pub(super) fn emit_finishes(
    sink: &mut dyn ProcessingSink,
    finishes: Option<&mut ModelFinishes<'_>>,
    meshes: &[MeshData],
    styles: &[StyleSource],
    styles_deferred: bool,
) {
    let Some(finishes) = finishes else { return };
    let batch: Vec<_> = meshes
        .iter()
        .zip(styles)
        .map(|(mesh, &style)| {
            let style = match mesh.geometry_item_id {
                Some(item) if styles_deferred => finishes.item_style_source(item).map_or(style, StyleSource::Style),
                _ => style,
            };
            finishes.finish_for_source(style)
        })
        .collect();
    sink.on_finishes(&batch);
}
