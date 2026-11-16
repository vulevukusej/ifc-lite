// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The empty-cut test on a plate (2 x 0.3 x 0.2 m, swept) placed turned and
//! moved, against box cutters placed relative to it.

use super::{CutVerdict, HostBody, EMPTY_CUT_TOLERANCE};
use crate::GeometryRouter;
use ifc_lite_core::EntityDecoder;

const PLATE: &str = "#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCDIRECTION((0.,0.,1.));
#3=IFCDIRECTION((1.,0.,0.));
#4=IFCAXIS2PLACEMENT3D(#1,#2,#3);
#5=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#4,$);
#6=IFCCARTESIANPOINT((120.5,-37.25,14.));
#7=IFCDIRECTION((0.6,0.,0.8));
#8=IFCDIRECTION((0.8,0.,-0.6));
#9=IFCAXIS2PLACEMENT3D(#6,#7,#8);
#10=IFCLOCALPLACEMENT($,#9);
#11=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,2.,0.3);
#12=IFCEXTRUDEDAREASOLID(#11,$,#2,0.2);
#13=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#12));
#14=IFCPRODUCTDEFINITIONSHAPE($,$,(#13));
#15=IFCPLATE('plate',$,$,$,$,#10,#14,$,$);
";

/// A box cutter `x` long, `y` wide, from `z0` up `z` high, centred at
/// (`cx`, 0) in the plate's frame, numbered from `base`.
fn cutter(base: u32, cx: f64, x: f64, y: f64, z0: f64, z: f64) -> String {
    format!(
        "#{a}=IFCCARTESIANPOINT(({cx:?},0.,{z0:?}));
#{b}=IFCAXIS2PLACEMENT3D(#{a},$,$);
#{c}=IFCLOCALPLACEMENT(#10,#{b});
#{d}=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{x:?},{y:?});
#{e}=IFCEXTRUDEDAREASOLID(#{d},$,#2,{z:?});
#{f}=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#{e}));
#{g}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{f}));
#{base}=IFCVOIDINGFEATURE('v{base}',$,$,$,$,#{c},#{g},$,.CUTOUT.);
",
        a = base + 1, b = base + 2, c = base + 3, d = base + 4, e = base + 5, f = base + 6, g = base + 7,
    )
}

fn verdicts(cutters: &[(u32, String)]) -> Vec<CutVerdict> {
    let content = format!("{PLATE}{}", cutters.iter().map(|(_, c)| c.as_str()).collect::<String>());
    let mut decoder = EntityDecoder::new(content.as_bytes());
    let router = GeometryRouter::new();
    let host = decoder.decode_by_id(15).expect("plate");
    let body = HostBody::of(&router, &host, &mut decoder).expect("a closed swept plate");
    cutters.iter().map(|(id, _)| body.judge(&router, *id, &mut decoder)).collect()
}

#[test]
fn a_cutter_flush_against_the_end_face_removes_nothing() {
    // x in [1, 1.5], wider and taller than the plate: it only touches the end face.
    let flush = cutter(100, 1.25, 0.5, 0.4, -0.1, 0.4);
    // The same, sunk into the plate by less than the tolerance.
    let sunk = cutter(200, 1.25 - 0.5 * EMPTY_CUT_TOLERANCE, 0.5, 0.4, -0.1, 0.4);
    assert_eq!(verdicts(&[(100, flush), (200, sunk)]), [CutVerdict::RemovesNothing; 2]);
}

#[test]
fn a_cutter_overlapping_by_a_millimetre_is_kept() {
    let overlap = cutter(100, 1.249, 0.5, 0.4, -0.1, 0.4);
    assert_eq!(verdicts(&[(100, overlap)]), [CutVerdict::CutsMaterial]);
}

#[test]
fn a_through_hole_in_a_swept_host_is_kept() {
    let hole = cutter(100, 0.0, 0.1, 0.05, -0.1, 0.4);
    assert_eq!(verdicts(&[(100, hole)]), [CutVerdict::CutsMaterial]);
}

#[test]
fn a_notch_flush_with_every_face_it_shares_is_kept() {
    // The last 0.1 m of the plate exactly: all its faces lie on the plate's
    // faces or inside it, so nothing crosses it, but it is solid material.
    let notch = cutter(100, 0.95, 0.1, 0.3, 0.0, 0.2);
    assert_eq!(verdicts(&[(100, notch)]), [CutVerdict::InsideHost]);
}

#[test]
fn a_cutter_clear_of_the_host_keeps_todays_handling() {
    let clear = cutter(100, 1.26, 0.5, 0.4, -0.1, 0.4);
    assert_eq!(verdicts(&[(100, clear)]), [CutVerdict::ClearOfHost]);
}

/// An L-shaped cutter (`IfcArbitraryClosedProfileDef`), its corner at the
/// plate's end face shifted by `dx`, extruded through the plate's thickness.
fn l_cutter(base: u32, dx: f64) -> String {
    let pts = [(1.0, -0.2), (1.5, -0.2), (1.5, 0.2), (1.3, 0.2), (1.3, 0.0), (1.0, 0.0)];
    let points: Vec<String> = pts
        .iter()
        .enumerate()
        .map(|(i, (x, y))| format!("#{}=IFCCARTESIANPOINT(({:?},{y:?}));", base + 10 + i as u32, x + dx))
        .collect();
    let refs: Vec<String> = (0..=pts.len()).map(|i| format!("#{}", base + 10 + (i % pts.len()) as u32)).collect();
    format!(
        "{}
#{a}=IFCPOLYLINE(({refs}));
#{b}=IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{a});
#{c}=IFCCARTESIANPOINT((0.,0.,-0.1));
#{d}=IFCAXIS2PLACEMENT3D(#{c},$,$);
#{e}=IFCLOCALPLACEMENT(#10,#{d});
#{f}=IFCEXTRUDEDAREASOLID(#{b},$,#2,0.4);
#{g}=IFCSHAPEREPRESENTATION(#5,'Body','SweptSolid',(#{f}));
#{h}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{g}));
#{base}=IFCVOIDINGFEATURE('l{base}',$,$,$,$,#{e},#{h},$,.NOTCH.);
",
        points.join("
"),
        refs = refs.join(","),
        a = base + 1, b = base + 2, c = base + 3, d = base + 4, e = base + 5, f = base + 6, g = base + 7, h = base + 8,
    )
}

#[test]
fn a_non_convex_cutter_flush_with_the_end_face_removes_nothing() {
    assert_eq!(verdicts(&[(100, l_cutter(100, 0.0))]), [CutVerdict::RemovesNothing]);
}

#[test]
fn a_non_convex_cutter_reaching_into_the_plate_is_kept() {
    assert_eq!(verdicts(&[(100, l_cutter(100, -0.001))]), [CutVerdict::CutsMaterial]);
}
