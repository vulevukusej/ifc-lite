// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A/B of the cut cache on one model: `cut_cache_probe <model.ifc> [quality]`.
//!
//! Runs `process` (full pass, `Low` quality unless `medium`/`high` is given)
//! with the cache off, then on, in one process, timing each; prints the
//! cache's counters, then compares the two outputs mesh by mesh. Meshes are
//! paired by express id and their order within the element. For a pair with
//! the same index buffer it reports the largest world-position difference
//! (origin folded in); for a pair whose triangulation differs it reports the
//! symmetric nearest-vertex distance instead. Colours and item ids must match.

use ifc_lite_geometry::{cut_cache_set_enabled_override, take_cut_cache_groups, take_cut_cache_stats};
use ifc_lite_processing::{process, MeshData, ProcessingRequest, ProcessingResult, TessellationQuality};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

/// Vertex pairs examined per element for the nearest-vertex distance.
const NEAREST_CAP: usize = 64_000_000;

fn run(bytes: &[u8], quality: TessellationQuality, cache: bool) -> ProcessingResult {
    cut_cache_set_enabled_override(Some(cache));
    let _ = (take_cut_cache_stats(), take_cut_cache_groups());
    let start = Instant::now();
    let result = process(bytes, &ProcessingRequest::full_pass().quality(quality), &mut ());
    let seconds = start.elapsed().as_secs_f64();
    let triangles: usize = result.meshes.iter().map(MeshData::triangle_count).sum();
    println!(
        "cache {}: {:.2} s, {} meshes, {} triangles",
        if cache { "on " } else { "off" },
        seconds,
        result.meshes.len(),
        triangles
    );
    println!("  {:?}", take_cut_cache_stats());
    result
}

fn world(mesh: &MeshData) -> Vec<[f64; 3]> {
    mesh.positions
        .chunks_exact(3)
        .map(|p| [mesh.origin[0] + p[0] as f64, mesh.origin[1] + p[1] as f64, mesh.origin[2] + p[2] as f64])
        .collect()
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Largest distance from a vertex of either mesh to the nearest vertex of the
/// other, or `None` above the cap.
fn nearest(a: &[[f64; 3]], b: &[[f64; 3]]) -> Option<f64> {
    if a.len().saturating_mul(b.len()) > NEAREST_CAP || a.is_empty() || b.is_empty() {
        return None;
    }
    let one_way = |from: &[[f64; 3]], to: &[[f64; 3]]| {
        from.iter()
            .map(|p| to.iter().map(|q| distance(p, q)).fold(f64::INFINITY, f64::min))
            .fold(0.0, f64::max)
    };
    Some(one_way(a, b).max(one_way(b, a)))
}

#[derive(Default)]
struct Element {
    express_id: u32,
    identical: bool,
    mesh_count_differs: bool,
    triangles_differ: bool,
    indices_differ: bool,
    attributes_differ: bool,
    same_topology_max: f64,
    nearest_max: Option<f64>,
    nearest_skipped: bool,
    volume_off: f64,
    volume_on: f64,
    area_off: f64,
    area_on: f64,
    triangles_off: usize,
    triangles_on: usize,
}

fn compare(id: u32, off: &[&MeshData], on: &[&MeshData]) -> Element {
    let mut element = Element { express_id: id, identical: true, ..Default::default() };
    if off.len() != on.len() {
        element.identical = false;
        element.mesh_count_differs = true;
        return element;
    }
    let reference = off.iter().chain(on).find_map(|m| world(m).first().copied()).unwrap_or([0.0; 3]);
    element.volume_off = off.iter().map(|m| volume(m, reference)).sum();
    element.volume_on = on.iter().map(|m| volume(m, reference)).sum();
    element.area_off = off.iter().map(|m| area(m)).sum();
    element.area_on = on.iter().map(|m| area(m)).sum();
    element.triangles_off = off.iter().map(|m| m.triangle_count()).sum();
    element.triangles_on = on.iter().map(|m| m.triangle_count()).sum();
    for (a, b) in off.iter().zip(on) {
        let same_bytes = a.positions == b.positions && a.indices == b.indices && a.normals == b.normals
            && a.origin == b.origin;
        if a.color != b.color || a.geometry_item_id != b.geometry_item_id || a.material_id != b.material_id {
            element.attributes_differ = true;
            element.identical = false;
        }
        if same_bytes {
            continue;
        }
        element.identical = false;
        if a.triangle_count() != b.triangle_count() {
            element.triangles_differ = true;
        }
        let (wa, wb) = (world(a), world(b));
        if a.indices == b.indices && wa.len() == wb.len() {
            let max = wa.iter().zip(&wb).map(|(p, q)| distance(p, q)).fold(0.0, f64::max);
            element.same_topology_max = element.same_topology_max.max(max);
        } else {
            element.indices_differ = true;
            match nearest(&wa, &wb) {
                Some(d) => element.nearest_max = Some(element.nearest_max.unwrap_or(0.0).max(d)),
                None => element.nearest_skipped = true,
            }
        }
    }
    element
}

fn by_element(meshes: &[MeshData]) -> BTreeMap<u32, Vec<&MeshData>> {
    let mut map: BTreeMap<u32, Vec<&MeshData>> = BTreeMap::new();
    for mesh in meshes {
        map.entry(mesh.express_id).or_default().push(mesh);
    }
    map
}

fn worst(elements: &[Element], key: impl Fn(&Element) -> f64, label: &str) {
    let mut ranked: Vec<&Element> = elements.iter().filter(|e| key(e) > 0.0).collect();
    ranked.sort_by(|a, b| key(b).total_cmp(&key(a)));
    let top: Vec<String> = ranked.iter().take(10).map(|e| format!("#{} {:.3e}", e.express_id, key(e))).collect();
    println!("  worst {label}: {}", top.join(", "));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: cut_cache_probe <model.ifc> [low|medium|high]");
    let quality = match args.get(1).map(String::as_str) {
        Some("medium") => TessellationQuality::Medium,
        Some("high") => TessellationQuality::High,
        _ => TessellationQuality::Low,
    };
    let bytes = std::fs::read(path).expect("read model");
    let off = run(&bytes, quality, false);
    let on = run(&bytes, quality, true);
    let groups = take_cut_cache_groups();
    cut_cache_set_enabled_override(None);

    let (off_map, on_map) = (by_element(&off.meshes), by_element(&on.meshes));
    let ids: Vec<u32> = off_map.keys().chain(on_map.keys()).copied().collect::<std::collections::BTreeSet<_>>()
        .into_iter().collect();
    let empty = Vec::new();
    let elements: Vec<Element> = ids
        .par_iter()
        .map(|id| compare(*id, off_map.get(id).unwrap_or(&empty), on_map.get(id).unwrap_or(&empty)))
        .collect();
    let differing: Vec<&Element> = elements.iter().filter(|e| !e.identical).collect();
    let count = |f: &dyn Fn(&Element) -> bool| elements.iter().filter(|e| f(e)).count();
    println!("elements: {}, byte-identical: {}, differing: {}", elements.len(),
        elements.len() - differing.len(), differing.len());
    println!("  mesh count differs: {}", count(&|e| e.mesh_count_differs));
    println!("  colour / item id / material differs: {}", count(&|e| e.attributes_differ));
    println!("  triangle count differs: {}", count(&|e| e.triangles_differ));
    println!("  index buffers differ: {}", count(&|e| e.indices_differ));
    println!("  nearest-vertex distance skipped (too large): {}", count(&|e| e.nearest_skipped));
    let rep_set: std::collections::HashSet<u32> = groups.iter().map(|g| g[0]).collect();
    println!("  representatives: {}, of them differing: {}", rep_set.len(),
        elements.iter().filter(|e| rep_set.contains(&e.express_id) && !e.identical).count());
    let changed: Vec<&Element> =
        elements.iter().filter(|e| e.indices_differ && (e.volume_on - e.volume_off).abs() > 1e-9).collect();
    let less = changed.iter().filter(|e| e.volume_on < e.volume_off).count();
    let max_dv = changed.iter().map(|e| (e.volume_on - e.volume_off).abs()).fold(0.0, f64::max);
    println!("  different indices, enclosed volume differs by > 1e-9 m3: {} (cache on removes more material: {}, less: {}), max {:.3e} m3",
        changed.len(), less, changed.len() - less, max_dv);
    let rel: Vec<&Element> = changed.iter().copied()
        .filter(|e| (e.volume_on - e.volume_off).abs() > 1e-4 * e.volume_off.abs().max(1e-12)).collect();
    println!("  of them by more than 1e-4 relative: {}", rel.len());
    worst(&elements, |e| (e.volume_on - e.volume_off).abs(), "volume difference (m3)");
    let same: Vec<&Element> = elements.iter().filter(|e| !e.identical && !e.indices_differ).collect();
    let same_rel = same.iter().map(|e| (e.volume_on - e.volume_off).abs() / e.volume_off.abs().max(1e-12))
        .fold(0.0, f64::max);
    println!("  same indices: {} elements, max relative volume difference {same_rel:.3e}", same.len());
    group_report(&groups, &elements);
    let same_max = elements.iter().map(|e| e.same_topology_max).fold(0.0, f64::max);
    let nearest_max = elements.iter().filter_map(|e| e.nearest_max).fold(0.0, f64::max);
    println!("  max world position difference, same indices: {same_max:.3e} m");
    println!("  max nearest-vertex distance, different indices: {nearest_max:.3e} m");
    worst(&elements, |e| e.same_topology_max, "same-index position difference");
    worst(&elements, |e| e.nearest_max.unwrap_or(0.0), "nearest-vertex distance");
    let listed: Vec<String> = differing.iter().filter(|e| e.mesh_count_differs || e.attributes_differ)
        .take(20).map(|e| format!("#{}", e.express_id)).collect();
    println!("  mesh-count or attribute differences: {}", listed.join(", "));
    let triangles: Vec<String> = differing.iter().filter(|e| e.triangles_differ)
        .take(20).map(|e| format!("#{}", e.express_id)).collect();
    println!("  triangle-count differences (first 20): {}", triangles.join(", "));
    let detail = std::env::var("CUT_PROBE_DETAIL").unwrap_or_default();
    for id in detail.split(',').filter_map(|id| id.trim().parse::<u32>().ok()) {
        let (a, b) = (off_map.get(&id).unwrap_or(&empty), on_map.get(&id).unwrap_or(&empty));
        for (x, y) in a.iter().zip(b).filter(|(x, y)| x.indices == y.indices) {
            let mut d: Vec<f64> = world(x).iter().zip(world(y)).map(|(p, q)| distance(p, &q)).collect();
            d.sort_by(f64::total_cmp);
            let over = d.iter().filter(|&&v| v > 1e-5).count();
            println!("  #{id} same-index vertex differences: n {} median {:.3e} p90 {:.3e} max {:.3e} over 1e-5: {over}",
                d.len(), d[d.len() / 2], d[d.len() * 9 / 10], d[d.len() - 1]);
        }
        for (label, map) in [("off", &off_map), ("on ", &on_map)] {
            for mesh in map.get(&id).unwrap_or(&empty) {
                println!("  #{id} {label}: {}", describe(mesh));
            }
        }
    }
}

/// Surface area of one mesh (invariant under a rigid move, closed or not).
fn area(mesh: &MeshData) -> f64 {
    let w = world(mesh);
    mesh.indices.chunks_exact(3).map(|t| {
        let [a, b, c] = [w[t[0] as usize], w[t[1] as usize], w[t[2] as usize]];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
    }).sum()
}

/// Signed enclosed volume of one mesh, its world coordinates taken relative
/// to `reference` so the triple products do not cancel at site scale.
fn volume(mesh: &MeshData, reference: [f64; 3]) -> f64 {
    let w: Vec<[f64; 3]> =
        world(mesh).into_iter().map(|p| [p[0] - reference[0], p[1] - reference[1], p[2] - reference[2]]).collect();
    mesh.indices.chunks_exact(3).map(|t| {
        let [a, b, c] = [w[t[0] as usize], w[t[1] as usize], w[t[2] as usize]];
        (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    }).sum()
}

/// Triangle count, world bounds, area and enclosed volume of one mesh.
fn describe(mesh: &MeshData) -> String {
    let w = world(mesh);
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in &w {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    let (mut area, mut volume) = (0.0, 0.0);
    for t in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [w[t[0] as usize], w[t[1] as usize], w[t[2] as usize]];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        area += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
    }
    format!("item {:?} tris {} min [{:.4},{:.4},{:.4}] max [{:.4},{:.4},{:.4}] area {:.6} volume {:.9}",
        mesh.geometry_item_id, mesh.triangle_count(), min[0], min[1], min[2], max[0], max[1], max[2], area, volume)
}

/// Per group: how far each served copy is from its representative (what
/// moving the cut preserves: triangle count and surface area), and how far
/// the copies' fresh cuts spread among themselves (what the uncached path
/// does by position alone).
fn group_report(groups: &[Vec<u32>], elements: &[Element]) {
    let by_id: std::collections::HashMap<u32, &Element> = elements.iter().map(|e| (e.express_id, e)).collect();
    let (mut moved_max, mut moved_tris, mut worst) = (0.0f64, 0usize, 0u32);
    let (mut spread_max, mut spread_groups, mut tri_groups) = (0.0f64, 0usize, 0usize);
    for group in groups {
        let Some(rep) = by_id.get(&group[0]) else { continue };
        let scale = rep.area_off.abs().max(1e-12);
        let (mut lo, mut hi, mut tris) = (f64::INFINITY, f64::NEG_INFINITY, std::collections::BTreeSet::new());
        for id in group {
            let Some(e) = by_id.get(id) else { continue };
            let moved = (e.area_on - rep.area_off).abs() / scale;
            if moved > moved_max {
                (moved_max, worst) = (moved, *id);
            }
            if e.triangles_on != rep.triangles_off {
                moved_tris += 1;
                println!("  served #{id} has {} triangles, its representative #{} {}", e.triangles_on, group[0],
                    rep.triangles_off);
            }
            lo = lo.min(e.area_off);
            hi = hi.max(e.area_off);
            tris.insert(e.triangles_off);
        }
        spread_groups += usize::from((hi - lo) / scale > 1e-4);
        tri_groups += usize::from(tris.len() > 1);
        spread_max = spread_max.max((hi - lo) / scale);
    }
    println!("  groups: {}; served copies vs their representative: triangle count differs {moved_tris},         max relative area difference {moved_max:.3e} (#{worst})", groups.len());
    println!("  fresh cuts within a group: max relative area spread {spread_max:.3e}, groups over 1e-4: {spread_groups},         groups with more than one triangle count: {tri_groups}");
}
