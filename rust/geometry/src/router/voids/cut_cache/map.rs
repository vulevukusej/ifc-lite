// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Moving a representative's cut onto a copy.
//!
//! The router places a body by its resolved, scaled placement `P` and then
//! subtracts the model RTC offset `c` (`transform_mesh_world`), so a vertex's
//! output position (`origin + position`) is `P·x − c`. A copy has the same
//! object-frame body `x`, so its vertex is `P_E·P_R⁻¹·(y_R + c) − c`: the
//! rotation `R` of `M = P_E·P_R⁻¹` and the translation `R·c + t_M − c`.
//! Normals turn by `R`; winding is kept because `det R > 0`.
//!
//! A mesh stored absolute (`origin` zero) stays absolute. A mesh stored
//! relative to an origin gets the moved origin snapped to the kernel grid, the
//! same snap `transform_mesh_world` gives a fresh origin, and its positions
//! relative to that.

use crate::kernel::mesh_bridge::SNAP_GRID;
use crate::math::mat4;
use crate::mesh::SubMesh;
use crate::router::transforms::FAR_RTC_FRAME_M;
use crate::{Mesh, SubMeshCollection};
use ifc_lite_core::{express_id::parse_express_id, step_encoding::step_string_literal_end, EntityDecoder};
use nalgebra::{Matrix3, Matrix4, Vector3};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Largest deviation of `RᵀR` from the identity, and of the homogeneous row
/// from `[0, 0, 0, 1]`, that still counts as rigid. Composed IFC placements
/// are orthonormal to ~1e-15; a scaled or sheared copy is off by far more.
pub(super) const RIGID_TOLERANCE: f64 = 1e-9;
/// Entity pairs walked to match a representative's item ids with a copy's.
const PAIR_BUDGET: usize = 4096;

/// The move from a representative's output frame to a copy's.
pub(crate) struct RigidMap {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    /// `M = P_E·P_R⁻¹`, for the placement matrices meshes carry.
    placement_map: Matrix4<f64>,
    rep_placement: [f64; 16],
    element_placement: [f64; 16],
}

/// Whether `m` is a proper rigid motion: orthonormal rotation, `det > 0`, and
/// an affine bottom row.
pub(super) fn is_rigid(m: &Matrix4<f64>) -> bool {
    let affine = m[(3, 0)].abs() <= RIGID_TOLERANCE
        && m[(3, 1)].abs() <= RIGID_TOLERANCE
        && m[(3, 2)].abs() <= RIGID_TOLERANCE
        && (m[(3, 3)] - 1.0).abs() <= RIGID_TOLERANCE;
    let r: Matrix3<f64> = m.fixed_view::<3, 3>(0, 0).into_owned();
    let gram = r.transpose() * r - Matrix3::identity();
    affine && m.iter().all(|v| v.is_finite()) && gram.amax() <= RIGID_TOLERANCE && r.determinant() > 0.0
}

impl RigidMap {
    /// The move from the representative placed by `rep` to the element placed
    /// by `element`, under RTC offset `rtc`; `None` unless it is rigid.
    pub(crate) fn between(rep: &Matrix4<f64>, element: &Matrix4<f64>, rtc: (f64, f64, f64)) -> Option<Self> {
        let placement_map = element * rep.try_inverse()?;
        if !is_rigid(&placement_map) {
            return None;
        }
        let rotation: Matrix3<f64> = placement_map.fixed_view::<3, 3>(0, 0).into_owned();
        let rtc = Vector3::new(rtc.0, rtc.1, rtc.2);
        let shift = Vector3::new(placement_map[(0, 3)], placement_map[(1, 3)], placement_map[(2, 3)]);
        Some(Self {
            rotation,
            translation: rotation * rtc + shift - rtc,
            placement_map,
            rep_placement: mat4::from_na(rep),
            element_placement: mat4::from_na(element),
        })
    }

    fn point(&self, p: Vector3<f64>) -> Vector3<f64> {
        self.rotation * p + self.translation
    }

    /// A row-major placement a mesh carries, moved to the copy: the copy's own
    /// placement when it is the representative's, else `M` composed onto it.
    fn placement(&self, row_major: &[f64; 16]) -> [f64; 16] {
        if *row_major == self.rep_placement {
            self.element_placement
        } else {
            mat4::from_na(&(self.placement_map * mat4::to_na(row_major)))
        }
    }
}

fn snap(v: f64) -> f64 {
    (v / SNAP_GRID).round() * SNAP_GRID
}

/// `mesh` moved by `map`, or `None` when a vertex is not finite or an absolute
/// mesh would land beyond the RTC distance at which the router stores a mesh
/// in a local frame (the copy is then cut afresh).
pub(super) fn move_mesh(mesh: &Mesh, map: &RigidMap) -> Option<Mesh> {
    let r = &map.rotation;
    let absolute = mesh.origin == [0.0; 3];
    let (origin, offset) = if absolute {
        ([0.0; 3], map.translation)
    } else {
        let moved = map.point(Vector3::new(mesh.origin[0], mesh.origin[1], mesh.origin[2]));
        let origin = [snap(moved.x), snap(moved.y), snap(moved.z)];
        (origin, moved - Vector3::new(origin[0], origin[1], origin[2]))
    };
    let mut positions = Vec::with_capacity(mesh.positions.len());
    for p in mesh.positions.chunks_exact(3) {
        let q = r * Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64) + offset;
        if !q.iter().all(|v| v.is_finite() && (!absolute || v.abs() < FAR_RTC_FRAME_M)) {
            return None;
        }
        positions.extend([q.x as f32, q.y as f32, q.z as f32]);
    }
    let mut normals = Vec::with_capacity(mesh.normals.len());
    for n in mesh.normals.chunks_exact(3) {
        let q = r * Vector3::new(n[0] as f64, n[1] as f64, n[2] as f64);
        normals.extend([q.x as f32, q.y as f32, q.z as f32]);
    }
    let instance_meta = mesh.instance_meta.clone().map(|mut meta| {
        meta.transform = map.placement(&meta.transform);
        meta
    });
    Some(Mesh {
        positions,
        normals,
        indices: mesh.indices.clone(),
        rtc_applied: mesh.rtc_applied,
        origin,
        instance_meta,
        local_bounds: mesh.local_bounds,
        local_to_world: mesh.local_to_world.as_ref().map(|m| map.placement(m)),
        welded_in_object_frame: mesh.welded_in_object_frame,
        // Kernel planes are in the representative's frame; `None` is always safe.
        plane_tags: None,
    })
}

/// `collection` moved by `map`, its item ids replaced through `ids`. `None`
/// for material-id sub-meshes, textured sub-meshes or an unpaired id.
pub(super) fn move_sub_meshes(collection: &SubMeshCollection, map: &RigidMap, ids: &FxHashMap<u32, u32>)
    -> Option<SubMeshCollection> {
    if collection.ids_are_materials {
        return None;
    }
    let mut sub_meshes = Vec::with_capacity(collection.sub_meshes.len());
    for sub in &collection.sub_meshes {
        if sub.uvs.is_some() || sub.texture.is_some() {
            return None;
        }
        sub_meshes.push(SubMesh::new(*ids.get(&sub.geometry_id)?, move_mesh(&sub.mesh, map)?));
    }
    Some(SubMeshCollection { sub_meshes, ids_are_materials: false })
}

/// The `#` references of a STEP record in attribute order, its own `#id=`
/// and anything inside a string literal skipped.
pub(super) fn record_refs(bytes: &[u8]) -> Vec<u32> {
    let start = bytes.iter().position(|&b| b == b'(').unwrap_or(bytes.len());
    let mut refs = Vec::new();
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => match step_string_literal_end(bytes, i) {
                Some(end) => i = end,
                None => break,
            },
            b'#' => {
                let digits = bytes[i + 1..].iter().take_while(|b| b.is_ascii_digit()).count();
                if let Some(id) = parse_express_id(&bytes[i + 1..i + 1 + digits]) {
                    refs.push(id);
                }
                i += 1 + digits;
            }
            _ => i += 1,
        }
    }
    refs
}

/// Pair the representative's entities under `rep_root` with the copy's under
/// `element_root`, breadth first, until every id in `wanted` is paired.
/// Structurally equal subtrees list their references in the same order, so
/// the n-th reference of a paired record pairs with the n-th of its partner.
/// `None` when a pair disagrees, the walk exceeds its budget, or an id is not
/// found.
pub(super) fn pair_ids(decoder: &mut EntityDecoder, rep_root: u32, element_root: u32, wanted: &[u32])
    -> Option<FxHashMap<u32, u32>> {
    let mut pairs: FxHashMap<u32, u32> = FxHashMap::default();
    let mut missing: Vec<u32> = wanted.to_vec();
    missing.sort_unstable();
    missing.dedup();
    let mut queue = VecDeque::from([(rep_root, element_root)]);
    while let Some((rep, element)) = queue.pop_front() {
        if missing.is_empty() {
            break;
        }
        if let Some(&paired) = pairs.get(&rep) {
            if paired != element {
                return None;
            }
            continue;
        }
        if pairs.len() >= PAIR_BUDGET {
            return None;
        }
        pairs.insert(rep, element);
        if let Ok(index) = missing.binary_search(&rep) {
            missing.remove(index);
        }
        let rep_refs = record_refs(decoder.get_raw_bytes(rep)?);
        let element_refs = record_refs(decoder.get_raw_bytes(element)?);
        if rep_refs.len() != element_refs.len() {
            return None;
        }
        queue.extend(rep_refs.into_iter().zip(element_refs));
    }
    missing.is_empty().then_some(pairs)
}
