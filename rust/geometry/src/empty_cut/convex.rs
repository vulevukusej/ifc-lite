// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Convex cutter pieces: face planes, clipping and ray parity.

use super::{Solid, CONVEX_EPSILON, P3};

/// A convex cutter as outward face planes (unit normal, offset).
pub(super) struct Convex {
    planes: Vec<(P3, f64)>,
    pub(super) centroid: P3,
    pub(super) lo: P3,
    pub(super) hi: P3,
}

impl Convex {
    pub(super) fn of(tris: &[[P3; 3]]) -> Option<Self> {
        let solid = Solid::new(tris.to_vec())?;
        let points: Vec<P3> = tris.iter().flatten().copied().collect();
        let centroid = points.iter().sum::<P3>() / points.len() as f64;
        let mut planes: Vec<(P3, f64)> = Vec::new();
        for [a, b, c] in tris {
            let n = (b - a).cross(&(c - a));
            if n.norm() < 1e-14 {
                continue;
            }
            let mut n = n.normalize();
            let mut d = n.dot(a);
            if n.dot(&centroid) > d {
                (n, d) = (-n, -d);
            }
            if !planes.iter().any(|(m, e)| m.dot(&n) > 1.0 - 1e-12 && (e - d).abs() < 1e-9) {
                planes.push((n, d));
            }
        }
        let convex = planes.len() >= 4 && planes.iter().all(|(n, d)| points.iter().all(|p| n.dot(p) <= d + CONVEX_EPSILON));
        convex.then_some(Self { planes, centroid, lo: solid.lo, hi: solid.hi })
    }

    /// Distance from `p` to the nearest face plane (positive inside).
    pub(super) fn depth(&self, p: &P3) -> f64 {
        self.planes.iter().map(|(n, d)| d - n.dot(p)).fold(f64::INFINITY, f64::min)
    }

    /// Area of `tri` inside the cutter with every face plane moved outward by
    /// `offset` (negative: inward).
    pub(super) fn clipped_area(&self, tri: &[P3; 3], offset: f64) -> f64 {
        let mut poly: Vec<P3> = tri.to_vec();
        for (n, d) in &self.planes {
            let limit = d + offset;
            let mut next = Vec::with_capacity(poly.len() + 2);
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                let (da, db) = (n.dot(&a) - limit, n.dot(&b) - limit);
                if da <= 0.0 {
                    next.push(a);
                }
                if (da < 0.0) != (db < 0.0) && da != db {
                    next.push(a + (b - a) * (da / (da - db)));
                }
            }
            poly = next;
            if poly.len() < 3 {
                return 0.0;
            }
        }
        let mut sum = P3::zeros();
        for i in 1..poly.len() - 1 {
            sum += (poly[i] - poly[0]).cross(&(poly[i + 1] - poly[0]));
        }
        0.5 * sum.norm()
    }
}

/// A right prism (a profile extruded square to it) split into one convex
/// prism per triangle of its bottom cap, or `None` for any other solid. The
/// caps are the two planes of one normal direction; every other face must be
/// parallel to that direction.
pub(super) fn right_prisms(tris: &[[P3; 3]]) -> Option<Vec<Convex>> {
    let faces: Vec<(P3, f64, &[P3; 3])> = tris
        .iter()
        .filter_map(|t| {
            let n = (t[1] - t[0]).cross(&(t[2] - t[0]));
            (n.norm() > 1e-14).then(|| (n.normalize(), 0.5 * n.norm(), t))
        })
        .collect();
    let mut axes: Vec<(P3, f64)> = Vec::new();
    for (n, area, _) in &faces {
        match axes.iter_mut().find(|(m, _)| m.dot(n).abs() > 1.0 - 1e-9) {
            Some((_, total)) => *total += area,
            None => axes.push((*n, *area)),
        }
    }
    axes.sort_by(|a, b| b.1.total_cmp(&a.1));
    'axis: for (axis, _) in axes.into_iter().take(3) {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut caps: Vec<(f64, &[P3; 3])> = Vec::new();
        for (n, _, t) in &faces {
            let c = n.dot(&axis).abs();
            if c > 1.0 - 1e-9 {
                let offset = axis.dot(&t[0]);
                lo = lo.min(offset);
                hi = hi.max(offset);
                caps.push((offset, t));
            } else if c > 1e-9 {
                continue 'axis;
            }
        }
        let height = hi - lo;
        if height <= 0.0 || caps.iter().any(|(o, _)| (o - lo).abs() > 1e-9 && (o - hi).abs() > 1e-9) {
            continue;
        }
        let shift = axis * height;
        let prisms: Option<Vec<Convex>> = caps
            .iter()
            .filter(|(o, _)| (o - lo).abs() <= 1e-9)
            .map(|(_, [a, b, c])| {
                let (d, e, f) = (a + shift, b + shift, c + shift);
                Convex::of(&[[*a, *b, *c], [d, e, f], [*a, *b, e], [*a, e, d], [*b, *c, f], [*b, f, e], [*c, *a, d], [*c, d, f]])
            })
            .collect();
        return prisms.filter(|p| !p.is_empty());
    }
    None
}

/// Ray-triangle crossings of the ray from `origin` along `dir` (Möller–Trumbore).
pub(super) fn crossings(solid: &Solid, origin: &P3, dir: &P3) -> usize {
    solid
        .tris
        .iter()
        .filter(|[a, b, c]| {
            let (e1, e2) = (b - a, c - a);
            let h = dir.cross(&e2);
            let det = e1.dot(&h);
            if det.abs() < 1e-18 {
                return false;
            }
            let f = 1.0 / det;
            let s = origin - a;
            let u = f * s.dot(&h);
            let q = s.cross(&e1);
            let v = f * dir.dot(&q);
            (0.0..=1.0).contains(&u) && v >= 0.0 && u + v <= 1.0 && f * e2.dot(&q) > 0.0
        })
        .count()
}
