// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Timing and byte-identity dump: `mesh_identity_dump <model.ifc> <runs> [out.txt]`.
//!
//! Runs `process` (full pass, `Low` quality) `runs` times, printing each wall
//! time and the median, then writes one line per mesh of the last run:
//! its index, express id and a SHA-256 over every field of `MeshData` (float
//! buffers by their bit patterns). Two builds produce byte-identical meshes
//! exactly when their dumps are equal; `diff` names the meshes that are not.

use ifc_lite_processing::{process, MeshData, ProcessingRequest, TessellationQuality};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::time::Instant;

fn floats(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for v in values {
        hasher.update(v.to_bits().to_le_bytes());
    }
}

fn debug(hasher: &mut Sha256, value: &impl std::fmt::Debug) {
    let text = format!("{value:?}");
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

fn mesh_hash(mesh: &MeshData) -> String {
    // Destructured exhaustively so a new field fails to compile here.
    let MeshData {
        express_id,
        ifc_type,
        global_id,
        name,
        presentation_layer,
        positions,
        normals,
        indices,
        color,
        material_name,
        geometry_item_id,
        material_id,
        properties,
        uvs,
        texture,
        geometry_class,
        origin,
        instance,
        local_bounds,
        local_to_world,
        presentation,
    } = mesh;
    let mut h = Sha256::new();
    h.update(express_id.to_le_bytes());
    debug(&mut h, ifc_type);
    debug(&mut h, global_id);
    debug(&mut h, name);
    debug(&mut h, presentation_layer);
    floats(&mut h, positions);
    floats(&mut h, normals);
    h.update((indices.len() as u64).to_le_bytes());
    for i in indices {
        h.update(i.to_le_bytes());
    }
    floats(&mut h, color);
    debug(&mut h, material_name);
    debug(&mut h, geometry_item_id);
    debug(&mut h, material_id);
    debug(&mut h, properties);
    match uvs {
        Some(uvs) => floats(&mut h, uvs),
        None => h.update(b"no-uvs"),
    }
    debug(&mut h, texture);
    h.update([*geometry_class]);
    for o in origin {
        h.update(o.to_bits().to_le_bytes());
    }
    debug(&mut h, instance);
    match local_bounds {
        Some(b) => floats(&mut h, b),
        None => h.update(b"no-bounds"),
    }
    match local_to_world {
        Some(m) => m.iter().for_each(|v| h.update(v.to_bits().to_le_bytes())),
        None => h.update(b"no-l2w"),
    }
    debug(&mut h, presentation);
    let mut out = String::with_capacity(64);
    for b in h.finalize() {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: mesh_identity_dump <model.ifc> <runs> [out.txt]");
    let runs: usize = args.next().map_or(1, |r| r.parse().expect("runs"));
    let out = args.next();
    let bytes = std::fs::read(&path).expect("read model");
    let request = ProcessingRequest::full_pass().quality(TessellationQuality::Low);
    let mut times = Vec::new();
    let mut last = None;
    for run in 0..runs {
        let start = Instant::now();
        let result = process(&bytes, &request, &mut ());
        let seconds = start.elapsed().as_secs_f64();
        let triangles: usize = result.meshes.iter().map(MeshData::triangle_count).sum();
        println!("run {run}: {seconds:.3} s, {} meshes, {triangles} triangles", result.meshes.len());
        times.push(seconds);
        last = Some(result);
    }
    times.sort_by(f64::total_cmp);
    println!("median {:.3} s of {:?}", times[times.len() / 2], times);
    if let (Some(out), Some(result)) = (out, last) {
        let mut text = String::new();
        let mut ids = std::collections::BTreeSet::new();
        for (i, mesh) in result.meshes.iter().enumerate() {
            ids.insert(mesh.express_id);
            let _ = writeln!(text, "{i} {} {}", mesh.express_id, mesh_hash(mesh));
        }
        let _ = writeln!(text, "instances {:?}", result.instances.len());
        std::fs::write(&out, text).expect("write dump");
        println!("wrote {} meshes of {} elements to {out}", result.meshes.len(), ids.len());
    }
}
