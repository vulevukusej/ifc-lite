// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Openings that remove nothing from their host.
//!
//! Some exporters write a host's body already cut (a tessellated net shape)
//! and still relate every cut to it as an `IfcVoidingFeature` or
//! `IfcOpeningElement`. Each such cutter sits flush against faces the body
//! already has, so subtracting it removes no material, but a boolean between
//! coincident faces is degenerate: its outcome depends on how each placed copy
//! happens to round, and it can chop off whole ends or leave slivers. This
//! module decides, before any cut, whether a cutter removes nothing, so the
//! caller can leave it out.
//!
//! The decision is made in the host's object frame, from the source items
//! meshed unplaced and the cutter moved by `P_host⁻¹ · P_opening` in f64, so
//! every copy of a host decides alike wherever it is placed. A cutter piece is
//! judged to remove nothing only when all of these hold, and otherwise it is
//! kept:
//!
//! - it is a convex solid (its vertices lie behind every one of its face
//!   planes), or a right prism (a profile extruded square to it, such as a
//!   cope's L) judged as one convex prism per triangle of its cap, and each
//!   convex piece's centroid lies deeper than `2ε` inside it;
//! - the host is one or more closed solids (every edge used exactly twice);
//! - no host triangle has area inside the cutter shrunk by `ε` (its face
//!   planes moved inward by [`EMPTY_CUT_TOLERANCE`]): the host's surface never
//!   passes through the cutter's interior;
//! - some host triangle has area inside the cutter grown by `ε`: the cutter
//!   touches the host (a cutter clear of the host keeps today's handling);
//! - the cutter's centroid is outside every host solid, by ray parity along
//!   three skew directions that must all agree.
//!
//! Since the host's surface does not cross the shrunk cutter, the shrunk
//! cutter's interior is wholly inside or wholly outside the host; the centroid
//! says which. So the volume a skipped cutter could have removed is at most
//! the `ε` shell inside its faces: `ε` times its surface area. `ε` is eight
//! cells of the kernel's snap grid (about 0.12 mm): a tessellated body cut by
//! its exporter sits within a few hundredths of a millimetre of the cutter it
//! was cut with, while a real hole, slot or cope goes millimetres deep.

use crate::kernel::mesh_bridge::SNAP_GRID;
use crate::representation::structural::{pds_representation_ids, product_shape_id, rep_item_ids};
use crate::{GeometryRouter, Mesh};
use ifc_lite_core::{DecodedEntity, EntityDecoder};
use nalgebra::{Matrix4, Point3, Vector3};
use rustc_hash::FxHashMap;

mod convex;
mod stats;
use convex::{crossings, right_prisms, Convex};
pub use stats::{record_empty_cut_verdict, skip_empty_cuts_enabled, take_empty_cut_stats, EmptyCutStats};

/// How far a host face may sit inside a cutter's face and still count as
/// flush with it, in metres: eight cells of the kernel's snap grid, 1/8192 m
/// (about 0.12 mm).
pub const EMPTY_CUT_TOLERANCE: f64 = 8.0 * SNAP_GRID;
/// Polygon area below which a clipped host triangle counts as nothing.
const AREA_EPSILON: f64 = 1e-12;
/// How far a cutter vertex may sit in front of one of its own face planes
/// and the cutter still count as convex.
const CONVEX_EPSILON: f64 = 1e-7;
/// Parity rays: skew, so no ray runs along an axis-aligned face or edge.
const RAYS: [[f64; 3]; 3] = [[0.8017, 0.4513, 0.3916], [-0.3261, 0.8813, -0.3420], [0.2672, -0.3107, 0.9122]];

type P3 = Vector3<f64>;

/// One closed solid as triangles in the host's object frame.
struct Solid {
    tris: Vec<[P3; 3]>,
    lo: P3,
    hi: P3,
}

impl Solid {
    fn new(tris: Vec<[P3; 3]>) -> Option<Self> {
        if tris.is_empty() {
            return None;
        }
        let mut lo = P3::repeat(f64::INFINITY);
        let mut hi = P3::repeat(f64::NEG_INFINITY);
        for p in tris.iter().flatten() {
            lo = lo.inf(p);
            hi = hi.sup(p);
        }
        Some(Self { tris, lo, hi })
    }
}

/// Triangles of `mesh` moved by `transform`, or `None` for a mesh that is
/// not in its plain object frame (RTC-rebased or carrying an origin).
fn triangles(mesh: &Mesh, transform: &Matrix4<f64>) -> Option<Vec<[P3; 3]>> {
    if mesh.rtc_applied || mesh.origin != [0.0; 3] {
        return None;
    }
    let point = |i: u32| -> Option<P3> {
        let p = mesh.positions.get(3 * i as usize..3 * i as usize + 3)?;
        Some(transform.transform_point(&Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)).coords)
    };
    mesh.indices.chunks_exact(3).map(|t| Some([point(t[0])?, point(t[1])?, point(t[2])?])).collect()
}

/// Every undirected edge (by its endpoints' bits) used exactly twice.
fn is_closed(mesh: &Mesh) -> bool {
    let key = |i: u32| -> Option<[u32; 3]> {
        let p = mesh.positions.get(3 * i as usize..3 * i as usize + 3)?;
        Some([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
    };
    let mut edges: FxHashMap<([u32; 3], [u32; 3]), u32> = FxHashMap::default();
    for t in mesh.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (Some(a), Some(b)) = (key(a), key(b)) else { return false };
            if a == b {
                continue;
            }
            *edges.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    !edges.is_empty() && edges.values().all(|&n| n == 2)
}

/// The unplaced meshes of an element's meshed representation items, in its
/// object frame (metres).
fn object_meshes(router: &GeometryRouter, element: &DecodedEntity, decoder: &mut EntityDecoder) -> Option<Vec<Mesh>> {
    let shape = decoder.decode_by_id(product_shape_id(element)?).ok()?;
    let reps: Vec<DecodedEntity> =
        pds_representation_ids(&shape).filter_map(|id| decoder.decode_by_id(id).ok()).collect();
    let mut meshes = Vec::new();
    for rep in crate::representation::meshed_representations(element, &reps) {
        for item_id in rep_item_ids(rep) {
            let item = decoder.decode_by_id(item_id).ok()?;
            let mesh = router.process_representation_item(&item, decoder).ok()?;
            if !mesh.is_empty() {
                meshes.push(mesh);
            }
        }
    }
    (!meshes.is_empty()).then_some(meshes)
}

fn placement(router: &GeometryRouter, element: &DecodedEntity, decoder: &mut EntityDecoder) -> Option<Matrix4<f64>> {
    let m = Matrix4::from_column_slice(&router.resolve_scaled_placement(element, decoder).ok()?);
    m.iter().all(|v| v.is_finite()).then_some(m)
}

/// Why a cutter was kept or skipped, for [`EmptyCutStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutVerdict {
    /// The cutter removes nothing: leave it out.
    RemovesNothing,
    /// The host's surface passes through the cutter's interior.
    CutsMaterial,
    /// The cutter's interior lies inside the host (it removes all of it).
    InsideHost,
    /// The cutter does not reach the host: today's handling is kept.
    ClearOfHost,
    /// The test does not apply (a non-convex or too thin cutter, an open or
    /// unreadable host or cutter): the cutter is kept.
    Undecided,
}

/// A host's body in its object frame, ready to judge each of its cutters.
pub struct HostBody {
    solids: Vec<Solid>,
    inverse_placement: Matrix4<f64>,
}

impl HostBody {
    /// The body of `host`, or `None` (keep every cutter) when it is not made
    /// of closed solids in a plain object frame.
    pub fn of(router: &GeometryRouter, host: &DecodedEntity, decoder: &mut EntityDecoder) -> Option<Self> {
        let inverse_placement = placement(router, host, decoder)?.try_inverse()?;
        let mut solids = Vec::new();
        for mesh in object_meshes(router, host, decoder)? {
            if !is_closed(&mesh) {
                return None;
            }
            solids.push(Solid::new(triangles(&mesh, &Matrix4::identity())?)?);
        }
        Some(Self { solids, inverse_placement })
    }

    /// Whether the opening `opening` removes nothing from this body.
    pub fn judge(&self, router: &GeometryRouter, opening: u32, decoder: &mut EntityDecoder) -> CutVerdict {
        let Some(pieces) = self.cutter(router, opening, decoder) else { return CutVerdict::Undecided };
        // An opening removes nothing only when every one of its pieces does.
        pieces
            .iter()
            .map(|piece| self.judge_piece(piece))
            .find(|verdict| *verdict != CutVerdict::RemovesNothing)
            .unwrap_or(CutVerdict::RemovesNothing)
    }

    /// The opening's item meshes in the host's object frame.
    fn cutter(&self, router: &GeometryRouter, opening: u32, decoder: &mut EntityDecoder) -> Option<Vec<Vec<[P3; 3]>>> {
        let entity = decoder.decode_by_id(opening).ok()?;
        let to_host = self.inverse_placement * placement(router, &entity, decoder)?;
        object_meshes(router, &entity, decoder)?.iter().map(|mesh| triangles(mesh, &to_host)).collect()
    }

    /// One cutter piece: convex, or a right prism split into convex prisms
    /// over its cap's triangles.
    fn judge_piece(&self, piece: &[[P3; 3]]) -> CutVerdict {
        if let Some(cutter) = Convex::of(piece) {
            return self.judge_convex(&cutter);
        }
        let Some(prisms) = right_prisms(piece) else { return CutVerdict::Undecided };
        // The pieces are judged alone; the cutter removes nothing when none
        // removes anything and at least one touches the host.
        let mut verdict = CutVerdict::ClearOfHost;
        for prism in &prisms {
            match self.judge_convex(prism) {
                CutVerdict::ClearOfHost => {}
                CutVerdict::RemovesNothing => verdict = CutVerdict::RemovesNothing,
                other => return other,
            }
        }
        verdict
    }

    fn judge_convex(&self, cutter: &Convex) -> CutVerdict {
        if cutter.depth(&cutter.centroid) <= 2.0 * EMPTY_CUT_TOLERANCE {
            return CutVerdict::Undecided;
        }
        let mut touches = false;
        for solid in &self.solids {
            if (0..3).any(|k| solid.hi[k] < cutter.lo[k] - EMPTY_CUT_TOLERANCE || solid.lo[k] > cutter.hi[k] + EMPTY_CUT_TOLERANCE) {
                continue;
            }
            for tri in &solid.tris {
                if (0..3).any(|k| {
                    tri.iter().all(|p| p[k] < cutter.lo[k] - EMPTY_CUT_TOLERANCE)
                        || tri.iter().all(|p| p[k] > cutter.hi[k] + EMPTY_CUT_TOLERANCE)
                }) {
                    continue;
                }
                if cutter.clipped_area(tri, -EMPTY_CUT_TOLERANCE) > AREA_EPSILON {
                    return CutVerdict::CutsMaterial;
                }
                touches = touches || cutter.clipped_area(tri, EMPTY_CUT_TOLERANCE) > AREA_EPSILON;
            }
        }
        if !touches {
            return CutVerdict::ClearOfHost;
        }
        let mut inside = [false; 3];
        for (k, ray) in RAYS.iter().enumerate() {
            inside[k] = self.solids.iter().any(|s| crossings(s, &cutter.centroid, &P3::from(*ray)) % 2 == 1);
        }
        match inside {
            [false, false, false] => CutVerdict::RemovesNothing,
            [true, true, true] => CutVerdict::InsideHost,
            _ => CutVerdict::Undecided,
        }
    }
}

#[cfg(test)]
mod tests;
