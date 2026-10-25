//! Evaluator: integrate recovered ΔT, Sim(3)/Umeyama alignment, metrics.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::path::Path;

#[cfg(feature = "serde")]
use std::fs;

pub use dynamics::Vec3;
pub use posebuilder::{Config as DynConfig, State as DynState};
pub use posebuilder::{PointingLaw, Pose};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sim3 {
    pub scale: f64,
    pub rotation: [[f64; 3]; 3],
    pub translation: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Metrics {
    pub rre_degrees: f64,
    pub rte_metres: f64,
    pub ate_metres: f64,
    pub rre: f64,
    pub rte: f64,
    pub ate: f64,
}

pub type Evaluation = Metrics;
pub type EvalResult = Metrics;
pub type Trajectory = Vec<Pose>;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

pub trait AsPose {
    fn as_pose(&self) -> Pose;
}
impl AsPose for Pose {
    fn as_pose(&self) -> Pose {
        *self
    }
}

pub trait Position {
    fn position(&self) -> Vec3;
}
impl Position for Vec3 {
    fn position(&self) -> Vec3 {
        *self
    }
}
impl Position for Pose {
    fn position(&self) -> Vec3 {
        self.translation
    }
}
impl Position for [f64; 3] {
    fn position(&self) -> Vec3 {
        Vec3::new(self[0], self[1], self[2])
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers (local)
// ---------------------------------------------------------------------------

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            if a[i][k] == 0.0 {
                continue;
            }
            let aik = a[i][k];
            for j in 0..3 {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn mat_vec(m: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    out
}

fn identity_matrix() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

pub fn integrate_deltas<D>(initial: &Pose, deltas: D) -> Vec<Pose>
where
    D: AsRef<[Pose]>,
{
    let deltas = deltas.as_ref();
    let mut out = Vec::with_capacity(deltas.len() + 1);
    let mut cur = *initial;
    out.push(cur);
    for d in deltas {
        cur = cur.compose(d);
        out.push(cur);
    }
    out
}

pub fn integrate_recovered(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}
pub fn integrate_sequence(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}
pub fn trajectory_from_deltas(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}
pub fn integrate_delta_sequence(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}

pub fn integrate_actions<A>(initial: &Pose, actions: A, k: f64) -> Vec<Pose>
where
    A: AsRef<[[f64; 9]]>,
{
    let actions = actions.as_ref();
    let deltas: Vec<Pose> = actions
        .iter()
        .map(|a| posebuilder::from_action(a, k))
        .collect();
    integrate_deltas(initial, &deltas)
}
pub fn integrate_recovered_actions(initial: &Pose, actions: &[[f64; 9]], k: f64) -> Vec<Pose> {
    integrate_actions(initial, actions, k)
}
pub fn trajectory_from_actions(initial: &Pose, actions: &[[f64; 9]], k: f64) -> Vec<Pose> {
    integrate_actions(initial, actions, k)
}

pub fn poses_to_positions(poses: &[Pose]) -> Vec<Vec3> {
    poses.iter().map(|p| p.translation).collect()
}
pub fn trajectory_positions(poses: &[Pose]) -> Vec<Vec3> {
    poses_to_positions(poses)
}
pub fn positions_of(poses: &[Pose]) -> Vec<Vec3> {
    poses_to_positions(poses)
}

// ---------------------------------------------------------------------------
// Umeyama / Sim(3) alignment
// ---------------------------------------------------------------------------

pub fn umeyama<S, T, P, Q>(src: S, dst: T) -> Sim3
where
    S: AsRef<[P]>,
    T: AsRef<[Q]>,
    P: Position,
    Q: Position,
{
    let src = src.as_ref();
    let dst = dst.as_ref();
    let n = src.len().min(dst.len());
    if n == 0 {
        return Sim3 {
            scale: 1.0,
            rotation: identity_matrix(),
            translation: Vec3::ZERO,
        };
    }
    let src_pts: Vec<Vec3> = (0..n).map(|i| src[i].position()).collect();
    let dst_pts: Vec<Vec3> = (0..n).map(|i| dst[i].position()).collect();

    let mut ms = Vec3::ZERO;
    let mut md = Vec3::ZERO;
    for i in 0..n {
        ms = ms.add(src_pts[i]);
        md = md.add(dst_pts[i]);
    }
    ms = ms.scale(1.0 / n as f64);
    md = md.scale(1.0 / n as f64);

    // centered
    let mut src_c = vec![Vec3::ZERO; n];
    let mut dst_c = vec![Vec3::ZERO; n];
    let mut src_var = 0.0;
    for i in 0..n {
        src_c[i] = src_pts[i].sub(ms);
        dst_c[i] = dst_pts[i].sub(md);
        src_var += src_c[i].dot(src_c[i]);
    }

    // H = sum src_c * dst_c^T  (3x3): H[i][j] = sum src_c_k[i] * dst_c_k[j]
    let mut h = [[0.0; 3]; 3];
    for k in 0..n {
        let s = src_c[k];
        let d = dst_c[k];
        h[0][0] += s.x * d.x;
        h[0][1] += s.x * d.y;
        h[0][2] += s.x * d.z;
        h[1][0] += s.y * d.x;
        h[1][1] += s.y * d.y;
        h[1][2] += s.y * d.z;
        h[2][0] += s.z * d.x;
        h[2][1] += s.z * d.y;
        h[2][2] += s.z * d.z;
    }

    let (u, _sig, v) = svd3(h);
    // R = V U^T
    let ut = transpose3(u);
    let r = mat_mul(v, ut);

    // Fix reflection
    let r = if det3(r) < 0.0 {
        // flip last column of V
        let mut v2 = v;
        for i in 0..3 {
            v2[i][2] = -v2[i][2];
        }
        mat_mul(v2, ut)
    } else {
        r
    };

    let scale = if src_var > 0.0 {
        // trace(Sigma) = sum over i of (U^T H V) diagonal; compute as trace(R H)? Use formula:
        // s = trace(Sigma) / src_var. trace(Sigma) = trace(U^T H V) = trace(H V U^T) ... simpler: sum singular values.
        // Recompute singular values from h and r? Use: s = trace(R H) / src_var? Actually optimal scale = tr(R^T H? no)
        // Standard: s = tr(Sigma) / sum||src_c||^2. Compute Sigma diag = (U^T H V)_ii.
        let uhv = mat_mul(mat_mul(transpose3(u), h), v);
        (uhv[0][0] + uhv[1][1] + uhv[2][2]) / src_var
    } else {
        1.0
    };

    // translation
    let r_scaled = scale_matrix(r, scale);
    let t = md.sub(mat_vec(r_scaled, ms));

    Sim3 {
        scale,
        rotation: r,
        translation: t,
    }
}

fn scale_matrix(m: [[f64; 3]; 3], s: f64) -> [[f64; 3]; 3] {
    let mut out = m;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] *= s;
        }
    }
    out
}

// aliases
pub fn sim3_alignment<S, T, P, Q>(src: S, dst: T) -> Sim3
where
    S: AsRef<[P]>,
    T: AsRef<[Q]>,
    P: Position,
    Q: Position,
{
    umeyama(src, dst)
}
pub fn umeyama_alignment<S, T, P, Q>(src: S, dst: T) -> Sim3
where
    S: AsRef<[P]>,
    T: AsRef<[Q]>,
    P: Position,
    Q: Position,
{
    umeyama(src, dst)
}
pub fn align_sim3<S, T, P, Q>(src: S, dst: T) -> Sim3
where
    S: AsRef<[P]>,
    T: AsRef<[Q]>,
    P: Position,
    Q: Position,
{
    umeyama(src, dst)
}
pub fn compute_sim3<S, T, P, Q>(src: S, dst: T) -> Sim3
where
    S: AsRef<[P]>,
    T: AsRef<[Q]>,
    P: Position,
    Q: Position,
{
    umeyama(src, dst)
}

pub fn apply_sim3<P>(sim: &Sim3, points: P) -> Vec<Vec3>
where
    P: AsRef<[Vec3]>,
{
    let points = points.as_ref();
    points
        .iter()
        .map(|p| {
            let rp = mat_vec(sim.rotation, *p);
            let scaled = rp.scale(sim.scale);
            scaled.add(sim.translation)
        })
        .collect()
}
pub fn align_points<P>(sim: &Sim3, points: P) -> Vec<Vec3>
where
    P: AsRef<[Vec3]>,
{
    apply_sim3(sim, points)
}
pub fn transform_points<P>(sim: &Sim3, points: P) -> Vec<Vec3>
where
    P: AsRef<[Vec3]>,
{
    apply_sim3(sim, points)
}
pub fn apply_alignment<P>(sim: &Sim3, points: P) -> Vec<Vec3>
where
    P: AsRef<[Vec3]>,
{
    apply_sim3(sim, points)
}

// ---------------------------------------------------------------------------
// SVD 3x3 (same as posebuilder)
// ---------------------------------------------------------------------------

fn svd3(m: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    let mut mtm = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += m[k][i] * m[k][j];
            }
            mtm[i][j] = s;
        }
    }

    let (v, evals) = jacobi_eigen(&mut mtm);

    let mut s = [0.0; 3];
    for i in 0..3 {
        s[i] = evals[i].max(0.0).sqrt();
    }

    let mut u = [[0.0; 3]; 3];
    for j in 0..3 {
        if s[j] > 1e-12 {
            for i in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += m[i][k] * v[k][j];
                }
                u[i][j] = acc / s[j];
            }
        } else {
            u[j][j] = 1.0;
        }
    }

    let detu = det3(u);
    let detv = det3(v);
    let (u, v, s) = if detu * detv < 0.0 {
        let mut u2 = u;
        let mut v2 = v;
        for i in 0..3 {
            u2[i][2] = -u2[i][2];
            v2[i][2] = -v2[i][2];
        }
        (u2, v2, s)
    } else {
        (u, v, s)
    };

    (u, [s[0].abs(), s[1].abs(), s[2].abs()], v)
}

fn jacobi_eigen(a: &mut [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut v = identity_matrix();
    let mut a = *a;
    for _ in 0..50 {
        let mut p = 0;
        let mut q = 1;
        let mut max = a[0][1].abs();
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > max {
                    max = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-12 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (theta * theta + 1.0).sqrt())
        } else {
            -1.0 / (-theta + (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..3 {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    let evals = [a[0][0], a[1][1], a[2][2]];
    (v, evals)
}

// ---------------------------------------------------------------------------
// Rotation helpers
// ---------------------------------------------------------------------------

fn rotation_angle_deg(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
    // angle of a * b^{-1}
    let bt = transpose3(b);
    let r = mat_mul(a, bt);
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    cos.acos() * 180.0 / PI
}

fn delta_pose(prev: &Pose, curr: &Pose) -> Pose {
    prev.inverse().compose(curr)
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

pub fn rre_degrees<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    let n = est.len().min(gt.len());
    if n < 2 {
        return 0.0;
    }
    let mut sumsq = 0.0;
    let mut count = 0usize;
    for i in 1..n {
        let de = delta_pose(&est[i - 1].as_pose(), &est[i].as_pose());
        let dg = delta_pose(&gt[i - 1].as_pose(), &gt[i].as_pose());
        let ang = rotation_angle_deg(de.rotation, dg.rotation);
        sumsq += ang * ang;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (sumsq / count as f64).sqrt()
    }
}

pub fn rte_metres<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    let n = est.len().min(gt.len());
    if n < 2 {
        return 0.0;
    }
    let mut sumsq = 0.0;
    let mut count = 0usize;
    for i in 1..n {
        let de = delta_pose(&est[i - 1].as_pose(), &est[i].as_pose());
        let dg = delta_pose(&gt[i - 1].as_pose(), &gt[i].as_pose());
        let diff = de.translation.sub(dg.translation).norm();
        sumsq += diff * diff;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (sumsq / count as f64).sqrt()
    }
}

pub fn ate_metres<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    let n = est.len().min(gt.len());
    if n == 0 {
        return 0.0;
    }
    let est_pos: Vec<Vec3> = (0..n).map(|i| est[i].as_pose().translation).collect();
    let gt_pos: Vec<Vec3> = (0..n).map(|i| gt[i].as_pose().translation).collect();
    let sim = umeyama(&est_pos, &gt_pos);
    let aligned = apply_sim3(&sim, &est_pos);
    let mut sumsq = 0.0;
    for i in 0..n {
        let diff = aligned[i].sub(gt_pos[i]).norm();
        sumsq += diff * diff;
    }
    (sumsq / n as f64).sqrt()
}

// aliases
pub fn relative_rotation_error<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    rre_degrees(est, gt)
}
pub fn relative_translation_error<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    rte_metres(est, gt)
}
pub fn absolute_trajectory_error<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    ate_metres(est, gt)
}
pub fn rre<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    rre_degrees(est, gt)
}
pub fn rte<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    rte_metres(est, gt)
}
pub fn ate<E, G, P>(est: E, gt: G) -> f64
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    ate_metres(est, gt)
}

pub fn evaluate<E, G, P>(est: E, gt: G) -> Metrics
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    let r = rre_degrees(est, gt);
    let t = rte_metres(est, gt);
    let a = ate_metres(est, gt);
    Metrics {
        rre_degrees: r,
        rte_metres: t,
        ate_metres: a,
        rre: r,
        rte: t,
        ate: a,
    }
}

pub fn compute_metrics<E, G, P>(est: E, gt: G) -> Metrics
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    evaluate(est, gt)
}
pub fn metrics<E, G, P>(est: E, gt: G) -> Metrics
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    evaluate(est, gt)
}
pub fn evaluate_trajectory<E, G, P>(est: E, gt: G) -> Metrics
where
    E: AsRef<[P]>,
    G: AsRef<[P]>,
    P: AsPose,
{
    let est = est.as_ref();
    let gt = gt.as_ref();
    evaluate(est, gt)
}

// ---------------------------------------------------------------------------
// Point-based metric variants (positions only)
// ---------------------------------------------------------------------------

pub fn ate_metres_points(est: &[Vec3], gt: &[Vec3]) -> f64 {
    let n = est.len().min(gt.len());
    if n == 0 {
        return 0.0;
    }
    let sim = umeyama(est, gt);
    let aligned = apply_sim3(&sim, est);
    let mut sumsq = 0.0;
    for i in 0..n {
        let diff = aligned[i].sub(gt[i]).norm();
        sumsq += diff * diff;
    }
    (sumsq / n as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dynamics::Vec3;

    #[test]
    fn test_mat_mul() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let res = mat_mul(a, b);
        assert_eq!(res, [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    }

    #[test]
    fn test_mat_vec() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(mat_vec(m, v), Vec3::new(1.0, 2.0, 3.0));
    }

    fn make_traj(n: usize, scale: f64, rot: f64, trans: Vec3) -> Vec<Pose> {
        // Build a trajectory where each step is same relative transform, then apply
        // a global sim3 to create misalignment case externally.
        let mut poses = Vec::with_capacity(n);
        let c = rot.cos();
        let s = rot.sin();
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut cur = Pose::identity();
        poses.push(cur);
        for _ in 1..n {
            let step = Pose {
                rotation: r,
                translation: Vec3::new(1.0, 0.0, 0.0).scale(scale),
            };
            cur = cur.compose(&step);
            poses.push(cur);
        }
        // apply global translation
        for p in &mut poses {
            p.translation = p.translation.add(trans);
        }
        poses
    }

    #[test]
    fn aligned_zero_error() {
        let traj = make_traj(10, 1.0, 0.1, Vec3::ZERO);
        let m = evaluate(&traj, &traj);
        assert!(m.ate_metres < 1e-9);
        assert!(m.rre_degrees < 1e-9);
        assert!(m.rte_metres < 1e-9);
    }

    #[test]
    fn sim3_aligned_zero_ate() {
        // est is gt transformed by a sim3 -> after alignment ATE ~0
        let gt = make_traj(12, 1.0, 0.05, Vec3::ZERO);
        // build est = scale 2, rotate 30deg, translate
        let ang: f64 = 0.3;
        let c = ang.cos();
        let s = ang.sin();
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let est: Vec<Pose> = gt
            .iter()
            .map(|p| {
                let rp = mat_vec(r, p.translation).scale(2.0);
                Pose {
                    rotation: mat_mul(r, p.rotation),
                    translation: rp.add(Vec3::new(5.0, -3.0, 2.0)),
                }
            })
            .collect();
        let m = evaluate(&est, &gt);
        assert!(m.ate_metres < 1e-6, "ATE {}", m.ate_metres);
    }

    #[test]
    fn misaligned_nonzero() {
        let a = make_traj(8, 1.0, 0.1, Vec3::ZERO);
        let mut b = make_traj(8, 1.0, 0.1, Vec3::ZERO);
        // perturb one pose
        b[3].translation = b[3].translation.add(Vec3::new(10.0, 0.0, 0.0));
        let m = evaluate(&a, &b);
        assert!(m.ate_metres > 0.0);
    }

    #[test]
    fn integrate_roundtrip() {
        let init = Pose::identity();
        let deltas = vec![
            Pose {
                rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                translation: Vec3::new(1.0, 0.0, 0.0),
            },
            Pose {
                rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                translation: Vec3::new(0.0, 1.0, 0.0),
            },
        ];
        let traj = integrate_deltas(&init, &deltas);
        assert_eq!(traj.len(), 3);
        assert!((traj[2].translation.x - 1.0).abs() < 1e-9);
        assert!((traj[2].translation.y - 1.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Evaluator struct wrapper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Evaluator {
    pub k: f64,
}

impl Evaluator {
    pub fn new(k: f64) -> Self {
        Self { k }
    }

    pub fn integrate<D>(&self, initial: &Pose, deltas: D) -> Vec<Pose>
    where
        D: AsRef<[Pose]>,
    {
        integrate_deltas(initial, deltas)
    }

    pub fn integrate_actions<A>(&self, initial: &Pose, actions: A) -> Vec<Pose>
    where
        A: AsRef<[[f64; 9]]>,
    {
        integrate_actions(initial, actions, self.k)
    }

    pub fn evaluate<E, G, P>(&self, est: E, gt: G) -> Metrics
    where
        E: AsRef<[P]>,
        G: AsRef<[P]>,
        P: AsPose,
    {
        evaluate(est, gt)
    }

    pub fn umeyama<S, T, P, Q>(&self, src: S, dst: T) -> Sim3
    where
        S: AsRef<[P]>,
        T: AsRef<[Q]>,
        P: Position,
        Q: Position,
    {
        umeyama(src, dst)
    }
}

// ---------------------------------------------------------------------------
// Convenience modules
// ---------------------------------------------------------------------------

pub mod metrics {
    pub use super::*;
}
pub mod alignment {
    pub use super::*;
}
pub mod trajectory {
    pub use super::*;
}
pub mod sim3 {
    pub use super::*;
}

// ---------------------------------------------------------------------------
// Optional serde support for Sim3
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct Sim3Serde {
    scale: f64,
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

#[cfg(feature = "serde")]
impl serde::Serialize for Sim3 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let helper = Sim3Serde {
            scale: self.scale,
            rotation: self.rotation,
            translation: [self.translation.x, self.translation.y, self.translation.z],
        };
        helper.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Sim3 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = Sim3Serde::deserialize(deserializer)?;
        Ok(Sim3 {
            scale: helper.scale,
            rotation: helper.rotation,
            translation: Vec3::new(
                helper.translation[0],
                helper.translation[1],
                helper.translation[2],
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors raised while loading job artifacts or evaluating a trajectory.
#[derive(Debug)]
pub enum EvalError {
    Io(String, String),
    Json(String),
    Invalid(String),
    Missing(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Io(path, err) => write!(f, "io error for {path}: {err}"),
            EvalError::Json(err) => write!(f, "invalid JSON: {err}"),
            EvalError::Invalid(msg) => write!(f, "invalid evaluator input: {msg}"),
            EvalError::Missing(msg) => write!(f, "missing evaluator input: {msg}"),
        }
    }
}

impl Error for EvalError {}

// ---------------------------------------------------------------------------
// Job I/O: sample_outputs.json / job.json / metrics.json
// ---------------------------------------------------------------------------

/// A recovered or reference sequence: an absolute `first_pose` plus the 9D
/// relative rows that follow it.  Integrate with [`Sequence::integrate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub first_pose: Pose,
    /// One 9D `[tx, ty, tz, r6]` row per frame transition.
    pub rows: Vec<[f64; 9]>,
    /// Kilometres per stored translation unit, applied only to translation.
    pub km_per_unit: f64,
}

impl Sequence {
    pub fn new(first_pose: Pose, rows: Vec<[f64; 9]>, km_per_unit: f64) -> Self {
        Self {
            first_pose,
            rows,
            km_per_unit,
        }
    }

    /// Number of frames (one more than the number of transitions).
    pub fn len(&self) -> usize {
        self.rows.len() + 1
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Reconstruct the absolute trajectory `T_t = T_{t-1} · ΔT_t`.
    pub fn integrate(&self) -> Vec<Pose> {
        integrate_actions(&self.first_pose, &self.rows, self.km_per_unit)
    }

    /// Same as [`Sequence::integrate`] but returns positions only.
    pub fn positions(&self) -> Vec<Vec3> {
        poses_to_positions(&self.integrate())
    }
}

/// Validate a list of raw 9D rows and copy them into fixed-size arrays.
pub fn validate_rows(rows: &[Vec<f64>], label: &str) -> Result<Vec<[f64; 9]>, EvalError> {
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        if row.len() != 9 {
            return Err(EvalError::Invalid(format!(
                "{label}[{i}] has {} values, expected 9",
                row.len()
            )));
        }
        let mut action = [0.0; 9];
        for (j, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(EvalError::Invalid(format!("{label}[{i}][{j}] is not finite")));
            }
            action[j] = *value;
        }
        out.push(action);
    }
    Ok(out)
}

/// `job.json` as written by `posebuilder`/`ipc`: the ground-truth pose sequence
/// lives in `first_pose` + `actions`.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JobManifest {
    pub schema: String,
    pub job_id: String,
    pub id: String,
    pub mode: String,
    pub model: String,
    pub scale_k_km_per_m: f64,
    pub pointing_law: String,
    pub fps: f64,
    pub frames: usize,
    pub action_dim: usize,
    pub action_encoding: String,
    pub frame_times_tdb_s: Vec<f64>,
    pub first_pose: Option<PoseJson>,
    pub actions: Vec<Vec<f64>>,
}

#[cfg(feature = "serde")]
impl Default for JobManifest {
    fn default() -> Self {
        Self {
            schema: String::new(),
            job_id: String::new(),
            id: String::new(),
            mode: String::new(),
            model: String::new(),
            scale_k_km_per_m: 1000.0,
            pointing_law: String::new(),
            fps: 24.0,
            frames: 0,
            action_dim: 9,
            action_encoding: String::new(),
            frame_times_tdb_s: Vec::new(),
            first_pose: None,
            actions: Vec::new(),
        }
    }
}

#[cfg(feature = "serde")]
impl JobManifest {
    /// Read `job.json` from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EvalError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| EvalError::Io(path.display().to_string(), e.to_string()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, EvalError> {
        serde_json::from_str(text).map_err(|e| EvalError::Json(e.to_string()))
    }

    /// `job_id` wins over `id` when both are present.
    pub fn effective_job_id(&self) -> &str {
        if !self.job_id.is_empty() {
            &self.job_id
        } else {
            &self.id
        }
    }

    /// True when the manifest carries a usable ground-truth pose sequence.
    pub fn has_actions(&self) -> bool {
        self.first_pose.is_some() && !self.actions.is_empty()
    }

    /// Ground-truth sequence from `first_pose` + `actions`.
    pub fn sequence(&self) -> Result<Sequence, EvalError> {
        let first_pose = self
            .first_pose
            .as_ref()
            .ok_or_else(|| EvalError::Missing("job.json has no first_pose".into()))?
            .to_pose();
        if self.actions.is_empty() {
            return Err(EvalError::Missing(
                "job.json has no actions to reconstruct a reference trajectory".into(),
            ));
        }
        let rows = validate_rows(&self.actions, "actions")?;
        let k = if self.scale_k_km_per_m.is_finite() && self.scale_k_km_per_m > 0.0 {
            self.scale_k_km_per_m
        } else {
            1000.0
        };
        Ok(Sequence::new(first_pose, rows, k))
    }
}

/// The absolute SE(3) transform serialised as `{rotation, translation_km}`.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseJson {
    pub rotation: [[f64; 3]; 3],
    pub translation_km: [f64; 3],
}

#[cfg(feature = "serde")]
impl PoseJson {
    pub fn to_pose(&self) -> Pose {
        Pose {
            rotation: self.rotation,
            translation: Vec3::new(
                self.translation_km[0],
                self.translation_km[1],
                self.translation_km[2],
            ),
        }
    }

    pub fn from_pose(pose: &Pose) -> Self {
        Self {
            rotation: pose.rotation,
            translation_km: [pose.translation.x, pose.translation.y, pose.translation.z],
        }
    }
}

/// `sample_outputs.json` as written by a Cosmos worker (see
/// `docs/cosmos-contract.md` §4.1).  `poses` is the recovered action sequence,
/// one 9D row per frame transition.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SampleOutputs {
    pub schema: String,
    pub job_id: String,
    pub mode: String,
    pub model: String,
    pub action_dim: usize,
    pub action_encoding: String,
    pub translation_unit: String,
    pub scale_k_km_per_m: f64,
    pub pointing_law: String,
    pub fps: f64,
    pub frames: usize,
    pub first_pose: Option<PoseJson>,
    pub poses: Vec<Vec<f64>>,
    pub mock: bool,
    pub note: Option<String>,
}

#[cfg(feature = "serde")]
impl Default for SampleOutputs {
    fn default() -> Self {
        Self {
            schema: String::new(),
            job_id: String::new(),
            mode: String::new(),
            model: String::new(),
            action_dim: 9,
            action_encoding: String::new(),
            translation_unit: String::from("cosmos_metre"),
            scale_k_km_per_m: 1000.0,
            pointing_law: String::new(),
            fps: 24.0,
            frames: 0,
            first_pose: None,
            poses: Vec::new(),
            mock: false,
            note: None,
        }
    }
}

#[cfg(feature = "serde")]
impl SampleOutputs {
    /// Read `sample_outputs.json` from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EvalError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| EvalError::Io(path.display().to_string(), e.to_string()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, EvalError> {
        serde_json::from_str(text).map_err(|e| EvalError::Json(e.to_string()))
    }

    /// Kilometres per stored translation unit.  Rows are normally in Cosmos
    /// metres, so the factor is `scale_k_km_per_m`; a literal `"km"` unit is a
    /// no-op.
    pub fn km_per_unit(&self) -> f64 {
        match self.translation_unit.as_str() {
            "km" => 1.0,
            _ if self.scale_k_km_per_m.is_finite() && self.scale_k_km_per_m > 0.0 => {
                self.scale_k_km_per_m
            }
            _ => 1000.0,
        }
    }

    pub fn validated_rows(&self) -> Result<Vec<[f64; 9]>, EvalError> {
        validate_rows(&self.poses, "poses")
    }

    /// True when the worker produced a pose sequence (false for `transfer`).
    pub fn has_poses(&self) -> bool {
        !self.poses.is_empty()
    }

    /// Build the recovered sequence.  `first_pose` falls back to the job
    /// manifest when the worker omitted it.
    pub fn sequence(&self, fallback_first_pose: Option<Pose>) -> Result<Sequence, EvalError> {
        let first_pose = match &self.first_pose {
            Some(pose) => pose.to_pose(),
            None => fallback_first_pose.ok_or_else(|| {
                EvalError::Missing(
                    "no first_pose in sample_outputs.json or job.json; cannot reconstruct poses"
                        .into(),
                )
            })?,
        };
        let rows = self.validated_rows()?;
        if rows.is_empty() {
            return Err(EvalError::Missing(
                "sample_outputs.json has an empty poses array; nothing to evaluate".into(),
            ));
        }
        Ok(Sequence::new(first_pose, rows, self.km_per_unit()))
    }
}

/// Sim(3) alignment parameters, serialised for the metrics report.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlignmentReport {
    pub scale: f64,
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

#[cfg(feature = "serde")]
impl AlignmentReport {
    pub fn from_sim3(sim: &Sim3) -> Self {
        Self {
            scale: sim.scale,
            rotation: sim.rotation,
            translation: [
                sim.translation.x,
                sim.translation.y,
                sim.translation.z,
            ],
        }
    }
}

/// The evaluator's output document (`cosmos-metrics/v1`).  Rotation error is
/// primary: translation in space is not metric-reliable, so ATE is reported
/// after a Sim(3)/Umeyama alignment.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricsReport {
    pub schema: String,
    pub job_id: String,
    pub mode: String,
    pub model: String,
    /// Path/description of the recovered `sample_outputs.json`.
    pub source: String,
    /// Path/description of the reference trajectory.
    pub reference: String,
    pub scale_k_km_per_m: f64,
    /// Frames compared (min of recovered and reference lengths).
    pub frames: usize,
    /// Recovered transitions integrated (before truncation to `frames`).
    pub pose_count: usize,
    pub rotation_primary: bool,
    pub aligned: bool,
    pub rre_degrees: f64,
    pub rte_metres: f64,
    pub ate_metres: f64,
    pub alignment: AlignmentReport,
    pub generated_by: String,
}

/// Align `recovered` against `reference` and compute RRE/RTE/ATE.
///
/// Both sequences are integrated from their own `first_pose`; if `recovered`
/// has no `first_pose` the reference's is used.  The returned report carries
/// provenance copied from `outputs` and the optional `job`.
#[cfg(feature = "serde")]
pub fn report_for(
    outputs: &SampleOutputs,
    job: Option<&JobManifest>,
    reference: &Sequence,
    reference_label: &str,
    source_label: &str,
) -> Result<MetricsReport, EvalError> {
    let fallback_first = job
        .and_then(|j| j.first_pose.as_ref())
        .map(|pose| pose.to_pose());
    let recovered = outputs.sequence(fallback_first)?;
    build_report(outputs, job, &recovered, reference, reference_label, source_label)
}

/// Like [`report_for`] but with a pre-built recovered [`Sequence`].
#[cfg(feature = "serde")]
pub fn build_report(
    outputs: &SampleOutputs,
    job: Option<&JobManifest>,
    recovered: &Sequence,
    reference: &Sequence,
    reference_label: &str,
    source_label: &str,
) -> Result<MetricsReport, EvalError> {
    let est = recovered.integrate();
    let gt = reference.integrate();
    let frames = est.len().min(gt.len());
    if frames < 2 {
        return Err(EvalError::Invalid(format!(
            "need at least 2 poses to evaluate (recovered={}, reference={})",
            est.len(),
            gt.len()
        )));
    }
    let est = &est[..frames];
    let gt = &gt[..frames];

    let metrics = evaluate(est, gt);
    let est_pos = poses_to_positions(est);
    let gt_pos = poses_to_positions(gt);
    let sim = umeyama(&est_pos, &gt_pos);

    let job_id = if !outputs.job_id.is_empty() {
        outputs.job_id.clone()
    } else {
        job.map(|j| j.effective_job_id().to_string())
            .unwrap_or_default()
    };
    let mode = non_empty(&outputs.mode)
        .or_else(|| job.map(|j| j.mode.clone()))
        .unwrap_or_default();
    let model = non_empty(&outputs.model)
        .or_else(|| job.map(|j| j.model.clone()))
        .unwrap_or_default();
    let scale_k = if outputs.scale_k_km_per_m.is_finite() && outputs.scale_k_km_per_m > 0.0 {
        outputs.scale_k_km_per_m
    } else {
        job.and_then(|j| {
            (j.scale_k_km_per_m.is_finite() && j.scale_k_km_per_m > 0.0)
                .then_some(j.scale_k_km_per_m)
        })
        .unwrap_or(1000.0)
    };

    Ok(MetricsReport {
        schema: METRICS_SCHEMA.to_string(),
        job_id,
        mode,
        model,
        source: source_label.to_string(),
        reference: reference_label.to_string(),
        scale_k_km_per_m: scale_k,
        frames,
        pose_count: recovered.rows.len(),
        rotation_primary: true,
        aligned: true,
        rre_degrees: metrics.rre_degrees,
        rte_metres: metrics.rte_metres,
        ate_metres: metrics.ate_metres,
        alignment: AlignmentReport::from_sim3(&sim),
        generated_by: "evaluator".to_string(),
    })
}

#[cfg(feature = "serde")]
fn non_empty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}

/// Metrics schema identifier written into [`MetricsReport::schema`].
pub const METRICS_SCHEMA: &str = "cosmos-metrics/v1";
/// `sample_outputs.json` schema identifier (see the contract §4.1).
pub const SAMPLE_OUTPUTS_SCHEMA: &str = "cosmos-sample-outputs/v1";

/// Serialise a report as pretty JSON.
#[cfg(feature = "serde")]
pub fn metrics_json(report: &MetricsReport) -> Result<String, EvalError> {
    serde_json::to_string_pretty(report).map_err(|e| EvalError::Json(e.to_string()))
}

/// Write a report to `path` (pretty JSON).
#[cfg(feature = "serde")]
pub fn write_metrics(
    path: impl AsRef<Path>,
    report: &MetricsReport,
) -> Result<(), EvalError> {
    let path = path.as_ref();
    let text = metrics_json(report)?;
    fs::write(path, text).map_err(|e| EvalError::Io(path.display().to_string(), e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests: JSON I/O and end-to-end evaluation
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "serde"))]
mod io_tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "schema": "cosmos-sample-outputs/v1",
        "job_id": "demo",
        "mode": "inverse_dynamics",
        "model": "Cosmos3-Nano",
        "action_dim": 9,
        "action_encoding": "translation_xyz + rot6d_first_two_columns",
        "translation_unit": "cosmos_metre",
        "scale_k_km_per_m": 1000.0,
        "fps": 24,
        "frames": 3,
        "first_pose": {
            "rotation": [[1,0,0],[0,1,0],[0,0,1]],
            "translation_km": [7000, 0, 0]
        },
        "poses": [
            [1.0, 0.0, 0.0, 1,0,0, 0,1,0],
            [1.0, 0.0, 0.0, 1,0,0, 0,1,0]
        ]
    }"#;

    #[test]
    fn parses_sample_outputs() {
        let outputs = SampleOutputs::from_json(SAMPLE).unwrap();
        assert_eq!(outputs.job_id, "demo");
        assert_eq!(outputs.poses.len(), 2);
        assert!(outputs.has_poses());
        let rows = outputs.validated_rows().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], 1.0);
    }

    #[test]
    fn integrates_recovered_trajectory() {
        let outputs = SampleOutputs::from_json(SAMPLE).unwrap();
        let seq = outputs.sequence(None).unwrap();
        // 2 transitions -> 3 frames.
        assert_eq!(seq.len(), 3);
        let traj = seq.integrate();
        // Each row moves 1 cosmos-metre * k(1000) = 1000 km along x.
        assert!((traj[0].translation.x - 7000.0).abs() < 1e-9);
        assert!((traj[2].translation.x - 9000.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_row_width() {
        let bad = r#"{"poses": [[1,2,3]]}"#;
        let outputs = SampleOutputs::from_json(bad).unwrap();
        assert!(matches!(
            outputs.validated_rows(),
            Err(EvalError::Invalid(_))
        ));
    }

    #[test]
    fn empty_poses_is_rejected() {
        let outputs = SampleOutputs::from_json(r#"{"poses": []}"#).unwrap();
        assert!(matches!(
            outputs.sequence(None),
            Err(EvalError::Missing(_))
        ));
    }

    #[test]
    fn reference_matches_recovered_gives_zero_error() {
        let outputs = SampleOutputs::from_json(SAMPLE).unwrap();
        let seq = outputs.sequence(None).unwrap();
        let report = report_for(&outputs, None, &seq, "self", "test").unwrap();
        assert!(report.ate_metres < 1e-9);
        assert!(report.rre_degrees < 1e-9);
        assert!(report.rte_metres < 1e-9);
        assert_eq!(report.schema, METRICS_SCHEMA);
        assert_eq!(report.frames, 3);
    }

    #[test]
    fn job_manifest_reconstructs_reference() {
        let job = JobManifest::from_json(
            r#"{
                "job_id": "demo",
                "first_pose": {"rotation": [[1,0,0],[0,1,0],[0,0,1]], "translation_km": [0,0,0]},
                "actions": [[1,0,0,1,0,0,0,1,0]],
                "scale_k_km_per_m": 1000.0
            }"#,
        )
        .unwrap();
        assert!(job.has_actions());
        let seq = job.sequence().unwrap();
        assert_eq!(seq.len(), 2);
        assert!((seq.integrate()[1].translation.x - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn sim3_transform_is_aligned_away() {
        // Non-collinear reference points: Umeyama is degenerate on a line.
        let gt: Vec<Vec3> = (0..6)
            .map(|i| {
                let a = i as f64 * 0.7;
                Vec3::new(a.cos() * 100.0, a.sin() * 80.0, i as f64 * 25.0)
            })
            .collect();

        let ang: f64 = 0.4;
        let (c, s) = (ang.cos(), ang.sin());
        let rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let scale = 3.0;
        let offset = Vec3::new(10.0, -5.0, 2.0);
        let est: Vec<Vec3> = gt
            .iter()
            .map(|p| mat_vec(rot, *p).scale(scale).add(offset))
            .collect();

        let m = ate_metres_points(&est, &gt);
        assert!(m < 1e-6, "ATE after alignment should vanish, got {m}");
    }
}
