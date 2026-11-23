// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The cut cache end to end: three copies of one voided plate, renumbered and
//! placed with different rotations and translations, meshed with the cache
//! off and on. The representative (lowest id) must come out byte-identical;
//! the two copies, served by moving its cut, must match their own fresh cuts
//! within 1e-6 relative (vertex sets, surface area and volume).
//!
//! One test in its own binary: the on/off switch is process-global.

use ifc_lite_geometry::{cut_cache_set_enabled_override, take_cut_cache_stats};
use ifc_lite_processing::{process, MeshData, ProcessingRequest, TessellationQuality};
use std::collections::BTreeMap;

const HEADER: &str = "ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('cut_cache_copies.ifc','2026-11-16T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCDIRECTION((1.,0.,0.));
#3=IFCDIRECTION((0.,0.,1.));
#4=IFCAXIS2PLACEMENT3D(#1,#3,#2);
#5=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#4,$);
#6=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#7=IFCUNITASSIGNMENT((#6));
#8=IFCPROJECT('0000000000000000000P01',$,'Project',$,$,$,$,(#5),#7);
#9=IFCLOCALPLACEMENT($,#4);
";

/// One plate (2 x 0.3 x 0.2 m) with a 0.2 x 0.1 m hole through it, written
/// with its own entity numbers from `base`, placed at `location` with `axis`
/// and `ref_dir`. The opening is placed relative to the plate's placement.
fn plate(base: u32, location: [f64; 3], axis: [f64; 3], ref_dir: [f64; 3]) -> String {
    let n = |offset: u32| base + offset;
    let v = |p: [f64; 3]| format!("({:?},{:?},{:?})", p[0], p[1], p[2]);
    let guid = |kind: char| format!("00000000000000000{kind}{base:04}");
    format!(
        "#{p0}=IFCCARTESIANPOINT({loc});
#{p1}=IFCDIRECTION({axis});
#{p2}=IFCDIRECTION({refd});
#{p3}=IFCAXIS2PLACEMENT3D(#{p0},#{p1},#{p2});
#{p4}=IFCLOCALPLACEMENT(#9,#{p3});
#{p5}=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,2.,0.3);
#{p6}=IFCEXTRUDEDAREASOLID(#{p5},$,#{p7},0.2);
#{p7}=IFCDIRECTION((0.,0.,1.));
#{p8}=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#{p6}));
#{p9}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{p8}));
#{p10}=IFCPLATE('{gid}',$,'Plate',$,$,#{p4},#{p9},$,$);
#{p11}=IFCCARTESIANPOINT((0.5,0.,-0.5));
#{p12}=IFCAXIS2PLACEMENT3D(#{p11},$,$);
#{p13}=IFCLOCALPLACEMENT(#{p4},#{p12});
#{p14}=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,0.2,0.1);
#{p15}=IFCEXTRUDEDAREASOLID(#{p14},$,#{p7},1.2);
#{p16}=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#{p15}));
#{p17}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{p16}));
#{p18}=IFCOPENINGELEMENT('{oid}',$,'Hole',$,$,#{p13},#{p17},$,.OPENING.);
#{p19}=IFCRELVOIDSELEMENT('{rid}',$,$,$,#{p10},#{p18});
",
        p0 = n(0), p1 = n(1), p2 = n(2), p3 = n(3), p4 = n(4), p5 = n(5), p6 = n(6), p7 = n(7),
        p8 = n(8), p9 = n(9), p10 = n(10), p11 = n(11), p12 = n(12), p13 = n(13), p14 = n(14),
        p15 = n(15), p16 = n(16), p17 = n(17), p18 = n(18), p19 = n(19),
        loc = v(location), axis = v(axis), refd = v(ref_dir),
        gid = guid('P'),
        oid = guid('O'),
        rid = guid('R'),
    )
}

fn model() -> String {
    let mut ifc = HEADER.to_string();
    ifc += &plate(100, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    // A quarter turn about Z, moved.
    ifc += &plate(200, [10.0, 5.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    // A general rotation: tilted about X, then turned about the new Z.
    ifc += &plate(300, [-4.0, 7.0, 3.0], [0.0, 0.6, 0.8], [0.6, 0.48, -0.36]);
    ifc += "ENDSEC;\nEND-ISO-10303-21;\n";
    ifc
}

const PLATES: [u32; 3] = [110, 210, 310];

fn meshes(bytes: &[u8], cache: bool) -> BTreeMap<u32, Vec<MeshData>> {
    cut_cache_set_enabled_override(Some(cache));
    let result = process(bytes, &ProcessingRequest::full_pass().quality(TessellationQuality::Low), &mut ());
    let mut by_id: BTreeMap<u32, Vec<MeshData>> = BTreeMap::new();
    for mesh in result.meshes {
        by_id.entry(mesh.express_id).or_default().push(mesh);
    }
    by_id
}

fn world(meshes: &[MeshData]) -> Vec<[f64; 3]> {
    meshes
        .iter()
        .flat_map(|m| {
            m.positions.chunks_exact(3).map(move |p| {
                [m.origin[0] + p[0] as f64, m.origin[1] + p[1] as f64, m.origin[2] + p[2] as f64]
            })
        })
        .collect()
}

/// Surface area and enclosed volume (divergence theorem) in world coordinates.
fn area_volume(meshes: &[MeshData]) -> (f64, f64) {
    let (mut area, mut volume) = (0.0, 0.0);
    for m in meshes {
        let w: Vec<[f64; 3]> = world(std::slice::from_ref(m));
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [w[t[0] as usize], w[t[1] as usize], w[t[2] as usize]];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
    }
    (area, volume.abs())
}

fn nearest(from: &[[f64; 3]], to: &[[f64; 3]]) -> f64 {
    from.iter()
        .map(|p| {
            to.iter()
                .map(|q| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

#[test]
fn copies_of_a_voided_plate_match_their_fresh_cuts() {
    let bytes = model().into_bytes();
    let _ = take_cut_cache_stats();
    let fresh = meshes(&bytes, false);
    let off = take_cut_cache_stats();
    assert_eq!(off.groups, 0, "cache off builds no groups: {off:?}");
    let cached = meshes(&bytes, true);
    let on = take_cut_cache_stats();
    cut_cache_set_enabled_override(None);
    assert_eq!((on.groups, on.grouped_hosts), (1, 3), "one group of three plates: {on:?}");
    assert!(on.served >= 2, "both copies served from the cache: {on:?}");
    assert_eq!(on.declined, 0, "{on:?}");

    let rep = PLATES[0];
    let (a, b) = (&fresh[&rep], &cached[&rep]);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!((&x.positions, &x.normals, &x.indices, x.origin), (&y.positions, &y.normals, &y.indices, y.origin),
            "the representative is byte-identical");
    }

    let (rep_area, rep_volume) = area_volume(a);
    // 2 x 0.3 x 0.2 minus a 0.2 x 0.1 x 0.2 hole.
    assert!((rep_volume - (0.12 - 0.004)).abs() < 1e-4, "the representative is cut: {rep_volume}");
    for id in &PLATES[1..] {
        let (a, b) = (&fresh[id], &cached[id]);
        assert_eq!(a.len(), b.len(), "#{id}: mesh count");
        for (x, y) in a.iter().zip(b) {
            assert_eq!((x.color, x.geometry_item_id), (y.color, y.geometry_item_id), "#{id}: attributes");
        }
        let (wa, wb) = (world(a), world(b));
        let scale = wa.iter().flatten().fold(1.0f64, |m, v| m.max(v.abs()));
        let gap = nearest(&wa, &wb).max(nearest(&wb, &wa));
        assert!(gap <= 1e-6 * scale, "#{id}: vertex sets {gap:e} apart");
        let ((area_a, volume_a), (area_b, volume_b)) = (area_volume(a), area_volume(b));
        assert!((area_a - area_b).abs() <= 1e-6 * area_a, "#{id}: area {area_a} vs {area_b}");
        assert!((volume_a - volume_b).abs() <= 1e-6 * volume_a, "#{id}: volume {volume_a} vs {volume_b}");
        assert!((area_b - rep_area).abs() <= 1e-6 * rep_area, "#{id}: same area as the representative");
    }
}
