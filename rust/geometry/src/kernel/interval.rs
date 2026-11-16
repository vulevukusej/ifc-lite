// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Directed-rounding f64 interval tier — the predicate cascade's fast path.
//!
//! Every op widens the interval OUTWARD (round-to-nearest then ±1 ULP), so the
//! true real value is always bracketed. A predicate's sign is returned only
//! when the result interval is strictly one side of zero; a straddling interval
//! is a genuine near-degeneracy and escalates to the exact (BigRational) tier.
//! Because the interval can never claim a definite sign it doesn't have, it can
//! never return a WRONG sign — proven by the soundness test against the oracle.
//!
//! `next_up`/`next_down` are integer bit-twiddles (not `f64::next_up`), so the
//! widening is bit-identical across x86_64/aarch64/wasm — the determinism bar.
//! No `mul_add`/FMA anywhere (contraction would break the directed rounding).

use super::{DropAxis, ImplicitPoint, Lpi, Sign, Tpi};

#[derive(Clone, Copy, Debug)]
pub struct RnInterval {
    pub lo: f64,
    pub hi: f64,
}

/// Smallest representable f64 strictly greater than `x` (toward +∞).
///
/// The common case steps the bit pattern by the sign bit alone (+1 for a
/// positive value, −1 for a negative one) without a data-dependent branch;
/// NaN, +∞ and the two zeros keep their own answers.
#[inline]
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1); // +smallest subnormal
    }
    let b = x.to_bits();
    let step = 1u64.wrapping_sub((b >> 63) << 1); // +1 positive, −1 negative
    f64::from_bits(b.wrapping_add(step))
}

/// Largest representable f64 strictly less than `x` (toward −∞). Branch-free
/// in the common case, as [`next_up`].
#[inline]
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1); // −smallest subnormal
    }
    let b = x.to_bits();
    let step = ((b >> 63) << 1).wrapping_sub(1); // −1 positive, +1 negative
    f64::from_bits(b.wrapping_add(step))
}

impl RnInterval {
    #[inline]
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    #[inline]
    pub fn add(self, o: Self) -> Self {
        Self { lo: next_down(self.lo + o.lo), hi: next_up(self.hi + o.hi) }
    }
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        Self { lo: next_down(self.lo - o.hi), hi: next_up(self.hi - o.lo) }
    }
    #[inline]
    pub fn mul(self, o: Self) -> Self {
        let (a, b, c, d) = (self.lo, self.hi, o.lo, o.hi);
        // Finite, ordered bounds: the extreme products follow from the signs
        // alone. Rounding to nearest is monotone, so the smallest (largest)
        // rounded product is the rounding of the smallest (largest) real
        // one, the value the four-product scan below selects; two such
        // values differ at most in the sign of a zero, which `next_down`
        // and `next_up` map to the same bound.
        if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() && a <= b && c <= d {
            let (mn, mx) = if a >= 0.0 {
                if c >= 0.0 {
                    (a * c, b * d)
                } else if d <= 0.0 {
                    (b * c, a * d)
                } else {
                    (b * c, b * d)
                }
            } else if b <= 0.0 {
                if c >= 0.0 {
                    (a * d, b * c)
                } else if d <= 0.0 {
                    (b * d, a * c)
                } else {
                    (a * d, a * c)
                }
            } else if c >= 0.0 {
                (a * d, b * d)
            } else if d <= 0.0 {
                (b * c, a * c)
            } else {
                let (ad, bc, ac, bd) = (a * d, b * c, a * c, b * d);
                (if bc < ad { bc } else { ad }, if bd > ac { bd } else { ac })
            };
            return Self { lo: next_down(mn), hi: next_up(mx) };
        }
        let c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi];
        let mut mn = c[0];
        let mut mx = c[0];
        for &v in &c[1..] {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        Self { lo: next_down(mn), hi: next_up(mx) }
    }
    /// Definite sign, or `None` if the interval straddles zero (escalate).
    #[inline]
    pub fn sign(self) -> Option<Sign> {
        if self.lo > 0.0 {
            Some(Sign::Positive)
        } else if self.hi < 0.0 {
            Some(Sign::Negative)
        } else if self.lo == 0.0 && self.hi == 0.0 {
            Some(Sign::Zero)
        } else {
            None
        }
    }
}

type Iv3 = [RnInterval; 3];

#[inline]
fn ivec(p: [f64; 3]) -> Iv3 {
    [RnInterval::point(p[0]), RnInterval::point(p[1]), RnInterval::point(p[2])]
}
#[inline]
fn isub(a: &Iv3, b: &Iv3) -> Iv3 {
    [a[0].sub(b[0]), a[1].sub(b[1]), a[2].sub(b[2])]
}
fn idet3(u: &Iv3, v: &Iv3, w: &Iv3) -> RnInterval {
    u[0]
        .mul(v[1].mul(w[2]).sub(v[2].mul(w[1])))
        .add(u[1].mul(v[2].mul(w[0]).sub(v[0].mul(w[2]))))
        .add(u[2].mul(v[0].mul(w[1]).sub(v[1].mul(w[0]))))
}

#[inline]
fn icross(u: &Iv3, v: &Iv3) -> Iv3 {
    [
        u[1].mul(v[2]).sub(u[2].mul(v[1])),
        u[2].mul(v[0]).sub(u[0].mul(v[2])),
        u[0].mul(v[1]).sub(u[1].mul(v[0])),
    ]
}

/// A small per-thread memo of interval lambdas, keyed by the defining
/// coordinates' bit patterns. The arrangement asks for the same implicit
/// point's lambda in many predicates (tri-tri ordering, seam splitting,
/// lexicographic order, point location) and the lambda is a pure function of
/// those coordinates, so a hit returns exactly what a recomputation would.
/// Direct-mapped and bounded; a collision simply recomputes.
struct LambdaMemo<const K: usize> {
    keys: Vec<[u64; K]>,
    values: Vec<Option<(Iv3, RnInterval)>>,
}

const MEMO_BITS: u32 = 10;

impl<const K: usize> LambdaMemo<K> {
    fn new() -> Self {
        Self { keys: vec![[0; K]; 1 << MEMO_BITS], values: vec![None; 1 << MEMO_BITS] }
    }

    fn slot(key: &[u64; K]) -> usize {
        let mut h = 0u64;
        for &w in key {
            h = (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
        (h >> (64 - MEMO_BITS)) as usize
    }
}

thread_local! {
    static LPI_MEMO: std::cell::RefCell<LambdaMemo<15>> = std::cell::RefCell::new(LambdaMemo::new());
    static TPI_MEMO: std::cell::RefCell<LambdaMemo<27>> = std::cell::RefCell::new(LambdaMemo::new());
}

fn memoised<const K: usize>(
    memo: &'static std::thread::LocalKey<std::cell::RefCell<LambdaMemo<K>>>,
    key: [u64; K],
    compute: impl FnOnce() -> (Iv3, RnInterval),
) -> (Iv3, RnInterval) {
    let slot = LambdaMemo::<K>::slot(&key);
    let hit = memo.with(|m| {
        let m = m.borrow();
        match m.values[slot] {
            Some(value) if m.keys[slot] == key => Some(value),
            _ => None,
        }
    });
    if let Some(value) = hit {
        return value;
    }
    let value = compute();
    memo.with(|m| {
        let mut m = m.borrow_mut();
        m.keys[slot] = key;
        m.values[slot] = Some(value);
    });
    value
}

fn lpi_lambda(l: &Lpi) -> (Iv3, RnInterval) {
    let mut key = [0u64; 15];
    for (k, c) in [l.p, l.q, l.r, l.s, l.t].iter().flatten().enumerate() {
        key[k] = c.to_bits();
    }
    memoised(&LPI_MEMO, key, || lpi_lambda_uncached(l))
}

fn lpi_lambda_uncached(l: &Lpi) -> (Iv3, RnInterval) {
    let p = ivec(l.p);
    let q = ivec(l.q);
    let rr = ivec(l.r);
    let s = ivec(l.s);
    let t = ivec(l.t);
    let qp = isub(&q, &p);
    let sr = isub(&s, &rr);
    let tr = isub(&t, &rr);
    let pr = isub(&p, &rr);
    let d = idet3(&qp, &sr, &tr);
    let n = idet3(&pr, &sr, &tr);
    // λ = d·P − n·(Q−P)  (see rational::lpi_lambda — the minus is load-bearing).
    let lx = d.mul(p[0]).sub(n.mul(qp[0]));
    let ly = d.mul(p[1]).sub(n.mul(qp[1]));
    let lz = d.mul(p[2]).sub(n.mul(qp[2]));
    ([lx, ly, lz], d)
}

fn tpi_lambda(t: &Tpi) -> (Iv3, RnInterval) {
    let mut key = [0u64; 27];
    for (k, c) in t.planes.iter().flatten().flatten().enumerate() {
        key[k] = c.to_bits();
    }
    memoised(&TPI_MEMO, key, || tpi_lambda_uncached(t))
}

fn tpi_lambda_uncached(t: &Tpi) -> (Iv3, RnInterval) {
    let plane = |pl: &[[f64; 3]; 3]| -> (Iv3, RnInterval) {
        let a = ivec(pl[0]);
        let ba = isub(&ivec(pl[1]), &a);
        let ca = isub(&ivec(pl[2]), &a);
        let n = icross(&ba, &ca);
        let off = n[0].mul(a[0]).add(n[1].mul(a[1])).add(n[2].mul(a[2]));
        (n, off)
    };
    let (n1, c1) = plane(&t.planes[0]);
    let (n2, c2) = plane(&t.planes[1]);
    let (n3, c3) = plane(&t.planes[2]);
    let d = idet3(&n1, &n2, &n3);
    let ns = [n1, n2, n3];
    let cs = [c1, c2, c3];
    let cramer = |k: usize| -> RnInterval {
        let mut rows = [ns[0], ns[1], ns[2]];
        for (row, ci) in rows.iter_mut().zip(cs.iter()) {
            row[k] = *ci;
        }
        idet3(&rows[0], &rows[1], &rows[2])
    };
    ([cramer(0), cramer(1), cramer(2)], d)
}

/// Interval-tier indirect orient3d for one implicit point `(λ/d)`. `None` ⇒ the
/// `Λ′` or `d` interval straddles zero ⇒ escalate to the exact tier.
pub(super) fn indirect_orient3d(
    lambda: &Iv3,
    d: RnInterval,
    p2: [f64; 3],
    p3: [f64; 3],
    p4: [f64; 3],
) -> Option<Sign> {
    let p4i = ivec(p4);
    let row1 = [
        lambda[0].sub(d.mul(p4i[0])),
        lambda[1].sub(d.mul(p4i[1])),
        lambda[2].sub(d.mul(p4i[2])),
    ];
    let row2 = isub(&ivec(p2), &p4i);
    let row3 = isub(&ivec(p3), &p4i);
    let lambda_det = idet3(&row1, &row2, &row3);
    let sd = d.sign()?;
    let sld = lambda_det.sign()?;
    Some(super::assemble_sign(sld, &[sd]))
}

pub fn lpi_orient3d(l: &Lpi, p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> Option<Sign> {
    let (lambda, d) = lpi_lambda(l);
    indirect_orient3d(&lambda, d, p2, p3, p4)
}

pub fn tpi_orient3d(t: &Tpi, p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> Option<Sign> {
    let (lambda, d) = tpi_lambda(t);
    indirect_orient3d(&lambda, d, p2, p3, p4)
}

#[inline]
fn axis_idx(axis: DropAxis) -> (usize, usize) {
    match axis {
        DropAxis::X => (1, 2),
        DropAxis::Y => (0, 2),
        DropAxis::Z => (0, 1),
    }
}

/// Interval-tier indirect orient2d for one implicit point `(λ/d)`. `None` ⇒ the
/// `Λ′₂` or `d` interval straddles zero ⇒ escalate to the exact tier.
fn indirect_orient2d(
    lambda: &Iv3,
    d: RnInterval,
    b: [f64; 3],
    c: [f64; 3],
    axis: DropAxis,
) -> Option<Sign> {
    let (i, j) = axis_idx(axis);
    let br = ivec(b);
    let cr = ivec(c);
    let li = lambda[i].sub(d.mul(cr[i]));
    let lj = lambda[j].sub(d.mul(cr[j]));
    let lambda_det2 = li.mul(br[j].sub(cr[j])).sub(lj.mul(br[i].sub(cr[i])));
    let sd = d.sign()?;
    let sld = lambda_det2.sign()?;
    Some(super::assemble_sign(sld, &[sd]))
}

pub fn lpi_orient2d(l: &Lpi, b: [f64; 3], c: [f64; 3], axis: DropAxis) -> Option<Sign> {
    let (lambda, d) = lpi_lambda(l);
    indirect_orient2d(&lambda, d, b, c, axis)
}

pub fn tpi_orient2d(t: &Tpi, b: [f64; 3], c: [f64; 3], axis: DropAxis) -> Option<Sign> {
    let (lambda, d) = tpi_lambda(t);
    indirect_orient2d(&lambda, d, b, c, axis)
}

/// Cached homogeneous interval lambda `(λ, d)` for ANY point — the f64-interval
/// analogue of `fixed::Lam`. Computed ONCE per interned point (the interner caches
/// it), so the hot re-triangulation predicates can run a directed-rounding f64
/// determinant straight from it — NO per-call LPI/TPI recompute, and crucially NO
/// wide-integer (I512) arithmetic, which wasm emulates ~hundreds× slower than
/// native. An `Explicit` point is `(coords, d=1)`.
pub(super) type IvLam = ([RnInterval; 3], RnInterval);

#[inline]
pub(super) fn ilambda_cached(p: &ImplicitPoint) -> IvLam {
    match p {
        ImplicitPoint::Lpi(l) => lpi_lambda(l),
        ImplicitPoint::Tpi(t) => tpi_lambda(t),
        ImplicitPoint::Explicit(e) => (ivec(*e), RnInterval::point(1.0)),
    }
}

/// 2-D orientation of three points from their CACHED interval lambdas — the exact
/// mirror of `fixed::orient2d_from_lam`'s determinant, in directed-rounding f64.
/// Returns `Some(sign)` ONLY when the interval determinant is strictly one side of
/// zero ⇒ that sign equals the exact sign (outward rounding, no FMA) and is
/// bit-identical native==wasm; a straddle returns `None` so the caller runs the
/// exact I512/BigRational tiers, unchanged.
pub(super) fn orient2d_from_lam_iv(a: &IvLam, b: &IvLam, c: &IvLam, axis: DropAxis) -> Option<Sign> {
    let (i, j) = axis_idx(axis);
    let (l1, d1) = (&a.0, a.1);
    let (l2, d2) = (&b.0, b.1);
    let (l3, d3) = (&c.0, c.1);
    let u_i = d1.mul(l2[i]).sub(d2.mul(l1[i]));
    let u_j = d1.mul(l2[j]).sub(d2.mul(l1[j]));
    let v_i = d1.mul(l3[i]).sub(d3.mul(l1[i]));
    let v_j = d1.mul(l3[j]).sub(d3.mul(l1[j]));
    let det = u_i.mul(v_j).sub(u_j.mul(v_i));
    Some(super::assemble_sign(det.sign()?, &[d2.sign()?, d3.sign()?]))
}

/// Interval tier of `cmp_along` — the f64 mirror of `fixed::cmp_along`: the sign
/// of `(a−b)·u` over the homogeneous lambdas. `None` on a zero-straddle ⇒ the
/// caller runs the exact I512/BigRational tiers. Closes the latent slow path where
/// EVERY tri-tri `cmp_along` skipped the float filter and went straight to bignum.
pub fn cmp_along(a: &ImplicitPoint, b: &ImplicitPoint, u: [f64; 3]) -> Option<Sign> {
    let (la, da) = ilambda_cached(a);
    let (lb, db) = ilambda_cached(b);
    let ur = ivec(u);
    let dot_a = la[0].mul(ur[0]).add(la[1].mul(ur[1])).add(la[2].mul(ur[2]));
    let dot_b = lb[0].mul(ur[0]).add(lb[1].mul(ur[1])).add(lb[2].mul(ur[2]));
    let num = dot_a.mul(db).sub(dot_b.mul(da));
    Some(super::assemble_sign(num.sign()?, &[da.sign()?, db.sign()?]))
}

/// Lexicographic compare of two points from their CACHED interval lambdas — the
/// f64 mirror of `fixed::cmp_lex_from_lam`. `None` as soon as any axis straddles.
pub(super) fn cmp_lex_from_lam_iv(a: &IvLam, b: &IvLam) -> Option<Sign> {
    let (la, da) = (&a.0, a.1);
    let (lb, db) = (&b.0, b.1);
    for k in 0..3 {
        let s = la[k].mul(db).sub(lb[k].mul(da));
        let sg = super::assemble_sign(s.sign()?, &[da.sign()?, db.sign()?]);
        if sg != Sign::Zero {
            return Some(sg);
        }
    }
    Some(Sign::Zero)
}

/// Interval tier of `rational::orient2d_2i` — `None` on a zero-straddle.
pub fn orient2d_2i(a: &ImplicitPoint, b: &ImplicitPoint, c: [f64; 3], axis: DropAxis) -> Option<Sign> {
    let (i, j) = axis_idx(axis);
    let (lam1, d1) = ilambda_cached(a);
    let (lam2, d2) = ilambda_cached(b);
    let cr = ivec(c);
    let a_i = lam1[i].sub(d1.mul(cr[i]));
    let a_j = lam1[j].sub(d1.mul(cr[j]));
    let b_i = lam2[i].sub(d2.mul(cr[i]));
    let b_j = lam2[j].sub(d2.mul(cr[j]));
    let det = a_i.mul(b_j).sub(a_j.mul(b_i));
    let sd1 = d1.sign()?;
    let sd2 = d2.sign()?;
    Some(super::assemble_sign(det.sign()?, &[sd1, sd2]))
}

/// Interval tier of `rational::orient2d_3i` — `None` on a zero-straddle.
pub fn orient2d_3i(a: &ImplicitPoint, b: &ImplicitPoint, c: &ImplicitPoint, axis: DropAxis) -> Option<Sign> {
    let (i, j) = axis_idx(axis);
    let (lam1, d1) = ilambda_cached(a);
    let (lam2, d2) = ilambda_cached(b);
    let (lam3, d3) = ilambda_cached(c);
    let u_i = d1.mul(lam2[i]).sub(d2.mul(lam1[i]));
    let u_j = d1.mul(lam2[j]).sub(d2.mul(lam1[j]));
    let v_i = d1.mul(lam3[i]).sub(d3.mul(lam1[i]));
    let v_j = d1.mul(lam3[j]).sub(d3.mul(lam1[j]));
    let det = u_i.mul(v_j).sub(u_j.mul(v_i));
    let sd2 = d2.sign()?;
    let sd3 = d3.sign()?;
    Some(super::assemble_sign(det.sign()?, &[sd2, sd3]))
}

/// One side of a `cmp_lex`: an explicit point's coordinates, or an implicit
/// point's interval lambda, computed once for all three axes.
enum LexSide {
    Explicit([f64; 3]),
    Implicit(IvLam),
}

impl LexSide {
    fn of(p: &ImplicitPoint) -> Self {
        match p {
            ImplicitPoint::Explicit(e) => Self::Explicit(*e),
            _ => Self::Implicit(ilambda_cached(p)),
        }
    }
}

/// Interval tier of one `cmp_lex` axis comparison — `None` on a zero-straddle.
fn icmp_axis(a: &LexSide, b: &LexSide, k: usize) -> Option<Sign> {
    use LexSide::{Explicit, Implicit};
    match (a, b) {
        (Explicit(ae), Explicit(be)) => Some(if ae[k] < be[k] {
            Sign::Negative
        } else if ae[k] > be[k] {
            Sign::Positive
        } else {
            Sign::Zero
        }),
        (Implicit((lam, d)), Explicit(be)) => {
            let num = lam[k].sub(d.mul(RnInterval::point(be[k])));
            Some(super::assemble_sign(num.sign()?, &[d.sign()?]))
        }
        (Explicit(ae), Implicit((lam, d))) => {
            let num = RnInterval::point(ae[k]).mul(*d).sub(lam[k]);
            Some(super::assemble_sign(num.sign()?, &[d.sign()?]))
        }
        (Implicit((la, da)), Implicit((lb, db))) => {
            let num = la[k].mul(*db).sub(lb[k].mul(*da));
            Some(super::assemble_sign(num.sign()?, &[da.sign()?, db.sign()?]))
        }
    }
}

/// Interval tier of `rational::cmp_lex`. `None` if the deciding axis straddles
/// zero (escalate to exact). A definite-Zero axis advances to the next.
pub fn cmp_lex(a: &ImplicitPoint, b: &ImplicitPoint) -> Option<Sign> {
    let (a, b) = (LexSide::of(a), LexSide::of(b));
    for k in 0..3 {
        match icmp_axis(&a, &b, k)? {
            Sign::Zero => continue,
            s => return Some(s),
        }
    }
    Some(Sign::Zero)
}

#[cfg(test)]
mod tests {
    use super::{next_down, next_up, RnInterval};

    /// The four-product scan `mul` used before its sign-case fast path.
    fn mul_by_scan(x: RnInterval, o: RnInterval) -> RnInterval {
        let c = [x.lo * o.lo, x.lo * o.hi, x.hi * o.lo, x.hi * o.hi];
        let mut mn = c[0];
        let mut mx = c[0];
        for &v in &c[1..] {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        RnInterval { lo: next_down(mn), hi: next_up(mx) }
    }

    /// The branch-free steps agree with the plain definition on every class
    /// of value.
    #[test]
    fn next_up_and_down_match_the_branching_definition() {
        let up = |x: f64| -> f64 {
            if x.is_nan() || x == f64::INFINITY {
                return x;
            }
            if x == 0.0 {
                return f64::from_bits(1);
            }
            let b = x.to_bits();
            f64::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
        };
        let down = |x: f64| -> f64 {
            if x.is_nan() || x == f64::NEG_INFINITY {
                return x;
            }
            if x == 0.0 {
                return -f64::from_bits(1);
            }
            let b = x.to_bits();
            f64::from_bits(if x > 0.0 { b - 1 } else { b + 1 })
        };
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut values = vec![
            0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -f64::NAN, f64::MAX, f64::MIN,
            f64::MIN_POSITIVE, -f64::MIN_POSITIVE, f64::from_bits(1), -f64::from_bits(1), 1.0, -1.0,
        ];
        for _ in 0..1_000_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(f64::from_bits(state));
        }
        for x in values {
            assert_eq!(next_up(x).to_bits(), up(x).to_bits(), "next_up({x:e})");
            assert_eq!(next_down(x).to_bits(), down(x).to_bits(), "next_down({x:e})");
        }
    }

    /// The sign-case fast path returns the scan's bounds bit for bit, for
    /// ordinary, tiny, huge, signed-zero, infinite, NaN and unordered bounds.
    #[test]
    fn mul_matches_the_four_product_scan_bit_for_bit() {
        let specials = [
            0.0, -0.0, 1.0, -1.0, f64::MIN_POSITIVE, -f64::MIN_POSITIVE, f64::from_bits(1),
            -f64::from_bits(1), f64::MAX, f64::MIN, 1.0e300, -1.0e300, f64::INFINITY,
            f64::NEG_INFINITY, f64::NAN, 3.0, -2.5, 0.1, -0.1, 1.0e-200,
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let value = |r: u64| -> f64 {
            match r % 4 {
                0 => specials[(r >> 8) as usize % specials.len()],
                1 => ((r >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1.0e4,
                2 => f64::from_bits(r.rotate_left(29)),
                _ => ((r >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0e-3,
            }
        };
        for _ in 0..2_000_000 {
            let (r1, r2, r3, r4) = (next(), next(), next(), next());
            let (a, b, c, d) = (value(r1), value(r2), value(r3), value(r4));
            // Mostly ordered, as the kernel builds them; sometimes not.
            let order = next() % 8 != 0;
            let (a, b) = if order && b < a { (b, a) } else { (a, b) };
            let (c, d) = if order && d < c { (d, c) } else { (c, d) };
            let x = RnInterval { lo: a, hi: b };
            let o = RnInterval { lo: c, hi: d };
            let fast = x.mul(o);
            let scan = mul_by_scan(x, o);
            assert_eq!(
                (fast.lo.to_bits(), fast.hi.to_bits()),
                (scan.lo.to_bits(), scan.hi.to_bits()),
                "[{a:e}, {b:e}] * [{c:e}, {d:e}]"
            );
        }
    }
}
