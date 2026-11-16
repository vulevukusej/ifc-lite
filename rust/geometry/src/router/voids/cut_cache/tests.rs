// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The cut cache's key, grouping, rigid check and move.

use super::key::{host_key, Hasher};
use super::map::{is_rigid, move_mesh, pair_ids, record_refs, RigidMap, RIGID_TOLERANCE};
use super::CutCache;
use crate::Mesh;
use ifc_lite_core::EntityDecoder;
use nalgebra::{Matrix4, Rotation3, Translation3, Vector3};

/// A plate with a hole, numbered from `base`: the plate's own placement
/// (`base+4`) at `location`, the hole's placement relative to it at `hole`.
/// `rel_to` overrides what the hole's placement is relative to; `depth` is
/// the plate's extrusion depth.
fn plate(base: u32, location: &str, hole: &str, rel_to: Option<u32>, depth: &str) -> String {
    let n = |offset: u32| base + offset;
    let rel_to = rel_to.unwrap_or(n(4));
    format!(
        "#{}=IFCCARTESIANPOINT({location});
#{}=IFCAXIS2PLACEMENT3D(#{},$,$);
#{}=IFCLOCALPLACEMENT(#9,#{});
#{}=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,2.,0.3);
#{}=IFCEXTRUDEDAREASOLID(#{},$,#3,{depth});
#{}=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#{}));
#{}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{}));
#{}=IFCPLATE('g{base}',$,'Plate',$,$,#{},#{},$,$);
#{}=IFCCARTESIANPOINT({hole});
#{}=IFCAXIS2PLACEMENT3D(#{},$,$);
#{}=IFCLOCALPLACEMENT(#{rel_to},#{});
#{}=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,0.2,0.1);
#{}=IFCEXTRUDEDAREASOLID(#{},$,#3,1.2);
#{}=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#{}));
#{}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{}));
#{}=IFCOPENINGELEMENT('o{base}',$,'Hole',$,$,#{},#{},$,.OPENING.);
",
        n(0), n(3), n(0), n(4), n(3), n(5), n(6), n(5), n(8), n(6), n(9), n(8), n(10), n(4), n(9),
        n(11), n(12), n(11), n(13), n(12), n(14), n(15), n(14), n(16), n(15), n(17), n(16), n(18), n(13), n(17),
    )
}

const PREAMBLE: &str = "#1=IFCCARTESIANPOINT((0.,0.,0.));
#3=IFCDIRECTION((0.,0.,1.));
#4=IFCAXIS2PLACEMENT3D(#1,$,$);
#5=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#4,$);
#9=IFCLOCALPLACEMENT($,#4);
";

/// Keys of the plates numbered from each base (host id `base+10`, opening
/// id `base+18`), all in one file.
fn keys(plates: &[String], bases: &[u32]) -> Vec<Option<u128>> {
    let content = format!("{PREAMBLE}{}", plates.concat());
    let mut decoder = EntityDecoder::new(content.as_bytes());
    let mut hasher = Hasher::new();
    bases.iter().map(|&b| host_key(&mut decoder, &mut hasher, b + 10, &[b + 18])).collect()
}

#[test]
fn copies_share_a_key_wherever_they_are_placed() {
    let plates = [
        plate(100, "(0.,0.,0.)", "(0.5,0.,-0.5)", None, "0.2"),
        plate(200, "(10.,5.,0.)", "(0.5,0.,-0.5)", None, "0.2"),
    ];
    let keys = keys(&plates, &[100, 200]);
    assert!(keys[0].is_some());
    assert_eq!(keys[0], keys[1], "renumbered, moved copies share a key");
}

#[test]
fn a_moved_cutter_or_another_body_changes_the_key() {
    let plates = [
        plate(100, "(0.,0.,0.)", "(0.5,0.,-0.5)", None, "0.2"),
        plate(200, "(0.,0.,0.)", "(0.6,0.,-0.5)", None, "0.2"),
        plate(300, "(0.,0.,0.)", "(0.5,0.,-0.5)", None, "0.25"),
    ];
    let keys = keys(&plates, &[100, 200, 300]);
    assert_ne!(keys[0], keys[1], "a cutter moved within the host");
    assert_ne!(keys[0], keys[2], "a different host body");
}

#[test]
fn an_opening_not_placed_relative_to_its_host_has_no_key() {
    // The hole's placement is relative to the world, not to the plate.
    let plates = [plate(100, "(0.,0.,0.)", "(0.5,0.,-0.5)", Some(9), "0.2")];
    assert_eq!(keys(&plates, &[100]), vec![None]);
}

#[test]
fn groups_take_their_lowest_id_as_representative() {
    let cache = CutCache::from_keys(&[(3, Some(7)), (5, Some(9)), (8, Some(7)), (11, None), (12, Some(7))]);
    assert_eq!(cache.group_count(), 1, "a key held by one host is no group");
    assert_eq!([3, 8, 12].map(|id| cache.representative(id)), [Some(3); 3]);
    assert_eq!(cache.representative(5), None);
    assert_eq!(cache.representative(11), None);
}

fn rotation(axis: Vector3<f64>, angle: f64) -> Matrix4<f64> {
    Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle).to_homogeneous()
}

#[test]
fn rigid_motions_pass_and_mirrors_and_scales_do_not() {
    let motion = Translation3::new(12.0, -3.5, 40.0).to_homogeneous() * rotation(Vector3::new(1.0, 2.0, 3.0), 0.7);
    assert!(is_rigid(&motion));
    assert!(is_rigid(&Matrix4::identity()));
    let mirror = Matrix4::new_nonuniform_scaling(&Vector3::new(-1.0, 1.0, 1.0));
    assert!(!is_rigid(&(motion * mirror)), "a mirror has det < 0");
    assert!(!is_rigid(&(motion * Matrix4::new_scaling(1.001))), "a scale");
    let mut shear = Matrix4::identity();
    shear[(0, 1)] = 10.0 * RIGID_TOLERANCE;
    assert!(!is_rigid(&shear), "a shear above the tolerance");
    shear[(0, 1)] = 0.1 * RIGID_TOLERANCE;
    assert!(is_rigid(&shear), "rounding below the tolerance");
    let mut projective = Matrix4::identity();
    projective[(3, 0)] = 0.5;
    assert!(!is_rigid(&projective));
}

#[test]
fn a_mirrored_copy_is_not_moved() {
    let rep = Translation3::new(1.0, 2.0, 3.0).to_homogeneous();
    let mirrored = rep * Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, -1.0, 1.0));
    assert!(RigidMap::between(&rep, &mirrored, (0.0, 0.0, 0.0)).is_none());
    assert!(RigidMap::between(&rep, &(rep * Matrix4::new_scaling(2.0)), (0.0, 0.0, 0.0)).is_none());
}

fn triangle(origin: [f64; 3], points: [[f32; 3]; 3]) -> Mesh {
    Mesh {
        positions: points.concat(),
        normals: [[0.0, 0.0, 1.0f32]; 3].concat(),
        indices: vec![0, 1, 2],
        origin,
        ..Mesh::default()
    }
}

fn world(mesh: &Mesh) -> Vec<Vector3<f64>> {
    mesh.positions
        .chunks_exact(3)
        .map(|p| Vector3::new(mesh.origin[0] + p[0] as f64, mesh.origin[1] + p[1] as f64, mesh.origin[2] + p[2] as f64))
        .collect()
}

/// The move reproduces `P_E·P_R⁻¹` in the RTC frame, for absolute and
/// origin-relative meshes, and turns normals by the rotation.
#[test]
fn the_move_matches_placing_the_copy_directly() {
    let rtc = (100.0, -50.0, 2.0);
    let rep = Translation3::new(103.0, -48.0, 5.0).to_homogeneous() * rotation(Vector3::z(), 0.3);
    let element = Translation3::new(90.0, -40.0, 7.0).to_homogeneous() * rotation(Vector3::new(0.2, 1.0, 0.4), 1.9);
    let map = RigidMap::between(&rep, &element, rtc).expect("rigid");
    let object = [Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.5, 0.0, 0.0), Vector3::new(0.0, 0.25, 0.0)];
    let rtc_v = Vector3::new(rtc.0, rtc.1, rtc.2);
    let output = |placement: &Matrix4<f64>, x: &Vector3<f64>| placement.transform_point(&(*x).into()).coords - rtc_v;
    let rep_out: Vec<Vector3<f64>> = object.iter().map(|x| output(&rep, x)).collect();
    let expected: Vec<Vector3<f64>> = object.iter().map(|x| output(&element, x)).collect();
    for origin in [[0.0; 3], [3.0, 2.0, 3.0]] {
        let o = Vector3::new(origin[0], origin[1], origin[2]);
        let points = [0, 1, 2].map(|i| {
            let p = rep_out[i] - o;
            [p.x as f32, p.y as f32, p.z as f32]
        });
        let moved = move_mesh(&triangle(origin, points), &map).expect("moved");
        for (got, want) in world(&moved).iter().zip(&expected) {
            assert!((got - want).norm() < 1e-5, "origin {origin:?}: {got:?} vs {want:?}");
        }
        let normal = Vector3::new(moved.normals[0] as f64, moved.normals[1] as f64, moved.normals[2] as f64);
        let want = element.fixed_view::<3, 3>(0, 0) * rep.fixed_view::<3, 3>(0, 0).transpose() * Vector3::z();
        assert!((normal - want).norm() < 1e-6);
        if origin != [0.0; 3] {
            assert!(moved.origin.iter().all(|v| (v * 65536.0).fract() == 0.0), "origin on the snap grid");
        }
    }
}

#[test]
fn the_move_keeps_the_copys_own_placement_exactly() {
    let rep = Translation3::new(1.0, 2.0, 3.0).to_homogeneous();
    let element = Translation3::new(-4.0, 8.0, 1.0).to_homogeneous() * rotation(Vector3::z(), 1.0);
    let map = RigidMap::between(&rep, &element, (0.0, 0.0, 0.0)).expect("rigid");
    let mut mesh = triangle([0.0; 3], [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0]]);
    mesh.local_to_world = Some(crate::math::mat4::from_na(&rep));
    let moved = move_mesh(&mesh, &map).expect("moved");
    assert_eq!(moved.local_to_world, Some(crate::math::mat4::from_na(&element)));
}

#[test]
fn record_refs_skip_the_own_id_and_strings() {
    assert_eq!(record_refs(b"#12=IFCSHAPEREPRESENTATION(#5,'Body #9','x',(#7,#8));"), vec![5, 7, 8]);
}

#[test]
fn item_ids_pair_through_equal_structure() {
    let plates = [
        plate(100, "(0.,0.,0.)", "(0.5,0.,-0.5)", None, "0.2"),
        plate(200, "(10.,5.,0.)", "(0.5,0.,-0.5)", None, "0.2"),
    ];
    let content = format!("{PREAMBLE}{}", plates.concat());
    let mut decoder = EntityDecoder::new(content.as_bytes());
    // Shape #109 holds representation #108 holding item #106.
    let pairs = pair_ids(&mut decoder, 109, 209, &[106]).expect("paired");
    assert_eq!(pairs.get(&106), Some(&206));
    assert!(pair_ids(&mut decoder, 109, 209, &[999]).is_none(), "an id outside the shape");
}
