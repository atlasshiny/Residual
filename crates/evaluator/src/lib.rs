//! Evaluator: integrate recovered ΔT, Sim(3)/Umeyama alignment, metrics.

use std::f64::consts::PI;

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
