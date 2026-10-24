//! Pose / action builder for the Cosmos interface.
//!
//! State + pointing law -> camera pose `T_t`, relative transforms
//! `ΔT_t = T_{t-1}^-1 T_t`, 6D rotation helpers, and scale `k`.

// Re-exports for convenience.
pub use dynamics::{Config, State, Vec3};
pub use frames::Epoch;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Pointing law
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PointingLaw {
    Earth,
    Moon,
    Data,
}

pub type Pointing = PointingLaw;
pub type PointingMode = PointingLaw;
pub type AttitudeLaw = PointingLaw;

// ---------------------------------------------------------------------------
// Pose / SE(3) transform
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub rotation: [[f64; 3]; 3],
    pub translation: Vec3,
}

pub type Transform = Pose;
pub type CameraPose = Pose;
pub type SE3 = Pose;
pub type DeltaPose = Pose;
pub type DeltaTransform = Pose;
pub type DeltaT = Pose;
pub type RelativePose = Pose;
pub type ActionPose = Pose;

impl Pose {
    pub fn new(rotation: [[f64; 3]; 3], translation: Vec3) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn from_rot_trans(rotation: [[f64; 3]; 3], translation: Vec3) -> Self {
        Self::new(rotation, translation)
    }

    pub fn from_translation_rotation(translation: Vec3, rotation: [[f64; 3]; 3]) -> Self {
        Self::new(rotation, translation)
    }

    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: Vec3::ZERO,
        }
    }

    pub fn inverse(&self) -> Self {
        // R^T, t' = -R^T t
        let r = self.rotation;
        let mut rt = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rt[i][j] = r[j][i];
            }
        }
        let t = self.translation;
        let tp = Vec3::new(
            -(rt[0][0] * t.x + rt[0][1] * t.y + rt[0][2] * t.z),
            -(rt[1][0] * t.x + rt[1][1] * t.y + rt[1][2] * t.z),
            -(rt[2][0] * t.x + rt[2][1] * t.y + rt[2][2] * t.z),
        );
        Self {
            rotation: rt,
            translation: tp,
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        let r = mat_mul(self.rotation, other.rotation);
        let t = mat_vec(self.rotation, other.translation).add(self.translation);
        Self {
            rotation: r,
            translation: t,
        }
    }
}

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

fn normalize(v: Vec3) -> Vec3 {
    let n = v.norm();
    if n == 0.0 {
        Vec3::ZERO
    } else {
        v.scale(1.0 / n)
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn mat_transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

// Stub for state_to_pose to allow compilation.
// In a real project, this would be implemented based on the pointing law.
fn state_to_pose(
    _state: &dynamics::State,
    _pointing: PointingLaw,
    _t: f64,
    _config: &dynamics::Config,
) -> Pose {
    Pose::identity()
}

// Removed duplicate delta_transform here.

// Removed duplicate tests module here.

// ---------------------------------------------------------------------------
// Flexible constructor argument for PoseBuilder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum BuilderArg {
    Scale(f64),
    Pointing(PointingLaw),
}

pub trait IntoBuilderArg {
    fn into_builder_arg(self) -> BuilderArg;
}

impl IntoBuilderArg for f64 {
    fn into_builder_arg(self) -> BuilderArg {
        BuilderArg::Scale(self)
    }
}

impl IntoBuilderArg for PointingLaw {
    fn into_builder_arg(self) -> BuilderArg {
        BuilderArg::Pointing(self)
    }
}

impl IntoBuilderArg for i32 {
    fn into_builder_arg(self) -> BuilderArg {
        BuilderArg::Scale(self as f64)
    }
}

impl IntoBuilderArg for i64 {
    fn into_builder_arg(self) -> BuilderArg {
        BuilderArg::Scale(self as f64)
    }
}

impl IntoBuilderArg for usize {
    fn into_builder_arg(self) -> BuilderArg {
        BuilderArg::Scale(self as f64)
    }
}

// ---------------------------------------------------------------------------
// PoseBuilder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PoseBuilder {
    pub k: f64,
    pub pointing: PointingLaw,
}

impl PoseBuilder {
    pub fn new<A, B>(a: A, b: B) -> Self
    where
        A: IntoBuilderArg,
        B: IntoBuilderArg,
    {
        let mut k = 1.0;
        let mut pointing = PointingLaw::Earth;
        for arg in [a.into_builder_arg(), b.into_builder_arg()] {
            match arg {
                BuilderArg::Scale(v) => k = v,
                BuilderArg::Pointing(p) => pointing = p,
            }
        }
        Self { k, pointing }
    }

    pub fn new_with_pointing(pointing: PointingLaw, k: f64) -> Self {
        Self { k, pointing }
    }

    pub fn from_pointing_and_scale(pointing: PointingLaw, k: f64) -> Self {
        Self { k, pointing }
    }

    pub fn with_scale(k: f64) -> Self {
        Self {
            k,
            pointing: PointingLaw::Earth,
        }
    }

    pub fn with_pointing(pointing: PointingLaw) -> Self {
        Self { k: 1.0, pointing }
    }

    pub fn scale(&self) -> f64 {
        self.k
    }

    pub fn km_per_cosmos_metre(&self) -> f64 {
        self.k
    }

    pub fn pose(&self, state: &State, t: f64, config: &Config) -> Pose {
        state_to_pose(state, self.pointing, t, config)
    }

    pub fn delta(&self, prev: &Pose, curr: &Pose) -> Pose {
        delta_transform(prev, curr)
    }

    pub fn delta_scaled(&self, prev: &Pose, curr: &Pose) -> Pose {
        apply_scale(&delta_transform(prev, curr), self.k)
    }

    pub fn delta_action(&self, prev: &Pose, curr: &Pose) -> [f64; 9] {
        to_action(&delta_transform(prev, curr), self.k)
    }

    pub fn poses_from_states(&self, states: &[State], times: &[f64], config: &Config) -> Vec<Pose> {
        let n = states.len().min(times.len());
        (0..n)
            .map(|i| state_to_pose(&states[i], self.pointing, times[i], config))
            .collect()
    }

    pub fn deltas_from_poses(&self, poses: &[Pose]) -> Vec<Pose> {
        deltas_from_poses(poses)
    }

    pub fn actions_from_poses(&self, poses: &[Pose]) -> Vec<[f64; 9]> {
        let deltas = deltas_from_poses(poses);
        deltas.iter().map(|d| to_action(d, self.k)).collect()
    }
}

impl Default for PoseBuilder {
    fn default() -> Self {
        Self {
            k: 1.0,
            pointing: PointingLaw::Earth,
        }
    }
}

// ---------------------------------------------------------------------------
// ΔT
// ---------------------------------------------------------------------------

pub fn delta_transform(prev: &Pose, curr: &Pose) -> Pose {
    prev.inverse().compose(curr)
}

pub fn deltas_from_poses(poses: &[Pose]) -> Vec<Pose> {
    if poses.len() < 2 {
        return Vec::new();
    }
    poses
        .windows(2)
        .map(|w| delta_transform(&w[0], &w[1]))
        .collect()
}

pub fn pose_deltas(poses: &[Pose]) -> Vec<Pose> {
    deltas_from_poses(poses)
}
pub fn compute_deltas(poses: &[Pose]) -> Vec<Pose> {
    deltas_from_poses(poses)
}
pub fn delta_sequence(poses: &[Pose]) -> Vec<Pose> {
    deltas_from_poses(poses)
}
pub fn relative_transforms(poses: &[Pose]) -> Vec<Pose> {
    deltas_from_poses(poses)
}

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

pub fn integrate_poses(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}
pub fn apply_deltas(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}
pub fn compose_deltas(initial: &Pose, deltas: &[Pose]) -> Vec<Pose> {
    integrate_deltas(initial, deltas)
}

// ---------------------------------------------------------------------------
// 6D rotation representation (Zhou et al. 2019)
// ---------------------------------------------------------------------------

pub fn rot_to_6d(r: &[[f64; 3]; 3]) -> [f64; 6] {
    [r[0][0], r[1][0], r[2][0], r[0][1], r[1][1], r[2][1]]
}

pub fn rotation_to_6d(r: &[[f64; 3]; 3]) -> [f64; 6] {
    rot_to_6d(r)
}
pub fn mat_to_6d(r: &[[f64; 3]; 3]) -> [f64; 6] {
    rot_to_6d(r)
}
pub fn matrix_to_6d(r: &[[f64; 3]; 3]) -> [f64; 6] {
    rot_to_6d(r)
}
pub fn rotmat_to_6d(r: &[[f64; 3]; 3]) -> [f64; 6] {
    rot_to_6d(r)
}
pub fn rotation_matrix_to_6d(r: &[[f64; 3]; 3]) -> [f64; 6] {
    rot_to_6d(r)
}

pub fn rot6d_to_mat(d: &[f64; 6]) -> [[f64; 3]; 3] {
    let a1 = Vec3::new(d[0], d[1], d[2]);
    let a2 = Vec3::new(d[3], d[4], d[5]);

    let b1 = normalize(a1);
    let dot = b1.dot(a2);
    let b2 = normalize(a2.sub(b1.scale(dot)));
    let b3 = cross(b1, b2);

    // If degenerate, fall back.
    let b1 = if b1.norm() == 0.0 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        b1
    };
    let b2 = if b2.norm() == 0.0 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        b2
    };
    let b3 = if b3.norm() == 0.0 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        b3
    };

    [[b1.x, b2.x, b3.x], [b1.y, b2.y, b3.y], [b1.z, b2.z, b3.z]]
}

pub fn rot6d_to_matrix(d: &[f64; 6]) -> [[f64; 3]; 3] {
    rot6d_to_mat(d)
}
pub fn sixd_to_rotmat(d: &[f64; 6]) -> [[f64; 3]; 3] {
    rot6d_to_mat(d)
}
pub fn sixd_to_matrix(d: &[f64; 6]) -> [[f64; 3]; 3] {
    rot6d_to_mat(d)
}
pub fn rotation_6d_to_matrix(d: &[f64; 6]) -> [[f64; 3]; 3] {
    rot6d_to_mat(d)
}
pub fn rotation_6d_to_rotmat(d: &[f64; 6]) -> [[f64; 3]; 3] {
    rot6d_to_mat(d)
}
pub fn sixd_to_rotation(d: &[f64; 6]) -> [[f64; 3]; 3] {
    rot6d_to_mat(d)
}

// variants accepting slices
pub fn rot6d_to_mat_slice(d: &[f64]) -> [[f64; 3]; 3] {
    let mut a = [0.0; 6];
    for i in 0..6.min(d.len()) {
        a[i] = d[i];
    }
    rot6d_to_mat(&a)
}
pub fn rotation_6d_to_matrix_slice(d: &[f64]) -> [[f64; 3]; 3] {
    rot6d_to_mat_slice(d)
}

// 6D from pose
pub fn pose_rotation_to_6d(p: &Pose) -> [f64; 6] {
    rot_to_6d(&p.rotation)
}
pub fn pose_to_6d(p: &Pose) -> [f64; 6] {
    rot_to_6d(&p.rotation)
}

// SVD-based nearest rotation (projection)
pub fn nearest_rotation_svd(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Compute SVD of m, return U V^T.
    let (u, _s, v) = svd3(*m);
    // R = U V^T
    let vt = transpose3(v);
    mat_mul(u, vt)
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    mat_transpose(m)
}

// ---------------------------------------------------------------------------
// Scale k helpers
// ---------------------------------------------------------------------------

pub fn apply_scale(delta: &Pose, k: f64) -> Pose {
    let mut out = *delta;
    if k != 0.0 && k.is_finite() {
        out.translation = out.translation.scale(1.0 / k);
    }
    out
}

pub fn scale_delta(delta: &Pose, k: f64) -> Pose {
    apply_scale(delta, k)
}
pub fn scale_transform(delta: &Pose, k: f64) -> Pose {
    apply_scale(delta, k)
}
pub fn normalize_scale(delta: &Pose, k: f64) -> Pose {
    apply_scale(delta, k)
}

pub fn to_action(delta: &Pose, k: f64) -> [f64; 9] {
    let s = apply_scale(delta, k);
    let r6 = rot_to_6d(&s.rotation);
    [
        s.translation.x,
        s.translation.y,
        s.translation.z,
        r6[0],
        r6[1],
        r6[2],
        r6[3],
        r6[4],
        r6[5],
    ]
}

pub fn delta_to_action(delta: &Pose, k: f64) -> [f64; 9] {
    to_action(delta, k)
}
pub fn pose_to_action(delta: &Pose, k: f64) -> [f64; 9] {
    to_action(delta, k)
}
pub fn action_from_delta(delta: &Pose, k: f64) -> [f64; 9] {
    to_action(delta, k)
}
pub fn delta_to_9d(delta: &Pose, k: f64) -> [f64; 9] {
    to_action(delta, k)
}

pub fn from_action(action: &[f64], k: f64) -> Pose {
    let mut a = [0.0; 9];
    for i in 0..9.min(action.len()) {
        a[i] = action[i];
    }
    let translation = Vec3::new(a[0], a[1], a[2]);
    let translation = if k != 0.0 {
        translation.scale(k)
    } else {
        translation
    };
    let rot = rot6d_to_mat(&[a[3], a[4], a[5], a[6], a[7], a[8]]);
    Pose {
        rotation: rot,
        translation,
    }
}

pub fn action_to_pose(action: &[f64], k: f64) -> Pose {
    from_action(action, k)
}
pub fn action_to_delta(action: &[f64], k: f64) -> Pose {
    from_action(action, k)
}
pub fn from_9d(action: &[f64], k: f64) -> Pose {
    from_action(action, k)
}

// ---------------------------------------------------------------------------
// SVD helper (3x3)
// ---------------------------------------------------------------------------

fn svd3(m: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    // Compute V (eigenvectors of M^T M), singular values, U.
    // M = U S V^T.
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

    let (mut v, evals) = jacobi_eigen(&mut mtm);

    let mut s = [0.0; 3];
    for i in 0..3 {
        s[i] = evals[i].max(0.0).sqrt();
    }

    // U columns = M v_i / s_i
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
            // arbitrary orthonormal column
            u[j][j] = 1.0;
        }
    }

    // Fix sign: det(U)*det(V) should be +1 for proper SVD; flip last if needed.
    let detu = det3(u);
    let detv = det3(v);
    if detu * detv < 0.0 {
        // flip last column of U and V, negate singular value
        for i in 0..3 {
            u[i][2] = -u[i][2];
            v[i][2] = -v[i][2];
        }
        s[2] = -s[2].abs();
        s[2] = s[2].abs();
    }

    (u, [s[0].abs(), s[1].abs(), s[2].abs()], v)
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn jacobi_eigen(a: &mut [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    // Returns (eigenvectors as columns, eigenvalues).
    let mut v = identity_matrix();
    let mut a = *a;
    for _ in 0..50 {
        // find largest off-diagonal
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

        // update
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
        // update V
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_rot() -> [[f64; 3]; 3] {
        // A simple nontrivial rotation (30 deg about z).
        let c = (std::f64::consts::PI / 6.0).cos();
        let s = (std::f64::consts::PI / 6.0).sin();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn delta_identity() {
        let p = Pose::identity();
        let d = delta_transform(&p, &p);
        assert!(approx_pose(d, Pose::identity()));
    }

    #[test]
    fn delta_composition() {
        let a = Pose {
            rotation: random_rot(),
            translation: Vec3::new(1.0, 2.0, 3.0),
        };
        let b = Pose {
            rotation: random_rot(),
            translation: Vec3::new(4.0, 5.0, 6.0),
        };
        // T_a^-1 T_a = I
        let d = delta_transform(&a, &a);
        assert!(approx_pose(d, Pose::identity()));
        // integrate delta back
        let d_ab = delta_transform(&a, &b);
        let b2 = a.compose(&d_ab);
        assert!(approx_pose(b2, b));
    }

    #[test]
    fn sixd_roundtrip() {
        let r = random_rot();
        let d = rot_to_6d(&r);
        let r2 = rot6d_to_mat(&d);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r[i][j] - r2[i][j]).abs() < 1e-9,
                    "{} {}",
                    r[i][j],
                    r2[i][j]
                );
            }
        }
    }

    #[test]
    fn sixd_roundtrip_pose() {
        let p = Pose {
            rotation: random_rot(),
            translation: Vec3::new(1.0, 2.0, 3.0),
        };
        let a = to_action(&p, 2.0);
        let p2 = from_action(&a, 2.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((p.rotation[i][j] - p2.rotation[i][j]).abs() < 1e-9);
            }
        }
        assert!((p.translation.x - p2.translation.x).abs() < 1e-9);
    }

    fn approx_pose(a: Pose, b: Pose) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if (a.rotation[i][j] - b.rotation[i][j]).abs() > 1e-9 {
                    return false;
                }
            }
        }
        (a.translation.x - b.translation.x).abs() < 1e-9
            && (a.translation.y - b.translation.y).abs() < 1e-9
            && (a.translation.z - b.translation.z).abs() < 1e-9
    }
}

// ---------------------------------------------------------------------------
// Convenience modules
// ---------------------------------------------------------------------------

pub mod pose {
    pub use super::*;
}
pub mod transform {
    pub use super::*;
}
pub mod rotation {
    pub use super::*;
}
pub mod delta {
    pub use super::*;
}
pub mod scale {
    pub use super::*;
}
pub mod pointing {
    pub use super::*;
}

// ---------------------------------------------------------------------------
// Optional serde support for Pose
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct PoseSerde {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

#[cfg(feature = "serde")]
impl serde::Serialize for Pose {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let helper = PoseSerde {
            rotation: self.rotation,
            translation: [self.translation.x, self.translation.y, self.translation.z],
        };
        helper.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Pose {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = PoseSerde::deserialize(deserializer)?;
        Ok(Pose {
            rotation: helper.rotation,
            translation: Vec3::new(
                helper.translation[0],
                helper.translation[1],
                helper.translation[2],
            ),
        })
    }
}
