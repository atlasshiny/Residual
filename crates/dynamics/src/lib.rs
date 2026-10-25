//! Minimal Rust core for the Earth-Moon restricted dynamics.
//!
//! Provides a fixed-step velocity-Verlet propagator ([`propagate`]) and an
//! adaptive Dormand-Prince (RK45) propagator ([`propagate_adaptive`]) built on
//! the [`dormand_prince_step`] embedded 4(5) error estimate.
//!
//! TODO: full SPICE ephemeris is not implemented yet.

use std::f64::consts::PI;

/// Allow `propagate` to accept any common integer step count.
pub trait IntoSteps {
    fn into_steps(self) -> usize;
}
impl IntoSteps for usize {
    fn into_steps(self) -> usize {
        self
    }
}
impl IntoSteps for u32 {
    fn into_steps(self) -> usize {
        self as usize
    }
}
impl IntoSteps for u64 {
    fn into_steps(self) -> usize {
        self as usize
    }
}
impl IntoSteps for i32 {
    fn into_steps(self) -> usize {
        self as usize
    }
}
impl IntoSteps for i64 {
    fn into_steps(self) -> usize {
        self as usize
    }
}
impl IntoSteps for isize {
    fn into_steps(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length(self) -> f64 {
        self.norm()
    }

    pub fn magnitude(self) -> f64 {
        self.norm()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs.scale(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    pub r: Vec3,
    pub v: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Burn {
    pub t: f64,
    pub dv: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub dt: f64,
    pub epoch: f64,
    pub mu_earth: f64,
    pub mu_moon: f64,
    pub earth_radius: f64,
    pub j2: f64,
    pub moon_distance: f64,
    pub moon_period: f64,
    pub moon_phase: f64,
    pub softening: f64,
    pub perturbation: i32,
    pub spin_axis: Vec3,
    pub burns: Vec<Burn>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 60.0,
            epoch: 0.0,
            mu_earth: K_MU_EARTH,
            mu_moon: K_MU_MOON,
            earth_radius: K_EARTH_RADIUS,
            j2: K_EARTH_J2,
            moon_distance: K_EARTH_MOON_DISTANCE,
            moon_period: K_MOON_PERIOD,
            moon_phase: 0.0,
            softening: 0.0,
            perturbation: 0,
            spin_axis: Vec3::new(0.0, 0.0, 1.0),
            burns: Vec::new(),
        }
    }
}

// Physical constants (SI), matching CUDA solver.
pub const K_PI: f64 = PI;
pub const K_MU_EARTH: f64 = 3.986004418e14;
pub const K_MU_MOON: f64 = 4.9048695e12;
pub const K_EARTH_RADIUS: f64 = 6.378137e6;
pub const K_EARTH_J2: f64 = 1.08262668e-3;
pub const K_EARTH_MOON_DISTANCE: f64 = 3.844e8;
pub const K_MOON_PERIOD: f64 = 2.3605915e6;

pub fn default_config() -> Config {
    Config::default()
}

/// Circular barycentric ephemeris matching the CUDA solver.
pub fn earth_moon_positions(t: f64, config: &Config) -> (Vec3, Vec3) {
    if config.moon_distance <= 0.0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }

    let mut theta = config.moon_phase;
    let total_mu = config.mu_earth + config.mu_moon;
    if config.moon_distance > 0.0 && total_mu > 0.0 {
        theta += (total_mu / (config.moon_distance * config.moon_distance * config.moon_distance))
            .sqrt()
            * t;
    }

    let moon_fraction = if total_mu > 0.0 {
        config.mu_moon / total_mu
    } else {
        0.0
    };

    let c = theta.cos();
    let s = theta.sin();

    let earth = Vec3::new(
        -config.moon_distance * moon_fraction * c,
        -config.moon_distance * moon_fraction * s,
        0.0,
    );
    let moon = Vec3::new(
        config.moon_distance * (1.0 - moon_fraction) * c,
        config.moon_distance * (1.0 - moon_fraction) * s,
        0.0,
    );
    (earth, moon)
}

/// J2 acceleration relative to Earth's centre.
pub fn earth_j2_acceleration(d: Vec3, config: &Config) -> Vec3 {
    let r2 = d.dot(d);
    if r2 <= 0.0 {
        return Vec3::ZERO;
    }
    let r = r2.sqrt();
    if r <= config.earth_radius {
        return Vec3::ZERO;
    }
    let s = config.spin_axis;
    let sdotr = d.x * s.x + d.y * s.y + d.z * s.z;
    let cos2 = (sdotr * sdotr) / r2;
    let k = -1.5 * config.j2 * config.mu_earth * config.earth_radius * config.earth_radius
        / (r2 * r2 * r);
    let factor = 1.0 - 5.0 * cos2;
    Vec3::new(
        k * (factor * d.x + 2.0 * sdotr * s.x),
        k * (factor * d.y + 2.0 * sdotr * s.y),
        k * (factor * d.z + 2.0 * sdotr * s.z),
    )
}

/// Acceleration with Earth/Moon point masses and optional J2.
pub fn acceleration(r: Vec3, config: &Config, t: f64) -> Vec3 {
    let (earth, moon) = earth_moon_positions(t, config);
    let d_earth = r.sub(earth);
    let d_moon = r.sub(moon);

    let eps2 = config.softening * config.softening;
    let r_earth2 = d_earth.dot(d_earth) + eps2;
    let r_moon2 = d_moon.dot(d_moon) + eps2;

    let mut a = Vec3::ZERO;
    if config.mu_earth > 0.0 {
        a = a.add(d_earth.scale(-config.mu_earth / (r_earth2 * r_earth2.sqrt())));
    }
    if config.mu_moon > 0.0 && config.moon_distance > 0.0 {
        a = a.add(d_moon.scale(-config.mu_moon / (r_moon2 * r_moon2.sqrt())));
    }
    if config.perturbation == 1 {
        a = a.add(earth_j2_acceleration(d_earth, config));
    }
    a
}

/// Fixed-step velocity-Verlet (kick-drift-kick) propagation, matching the CUDA solver. Returns trajectory including initial state.
pub fn propagate(state: State, steps: impl IntoSteps, dt: f64, config: &Config) -> Vec<State> {
    let steps = steps.into_steps();
    let mut out = Vec::with_capacity(steps + 1);
    // Velocity-Verlet (kick-drift-kick), matching the CUDA solver.
    let mut s = state;
    let mut t = config.epoch;
    out.push(s);
    for _ in 0..steps {
        // Apply impulsive burns firing at this step's start time.
        for burn in &config.burns {
            if (t - burn.t).abs() < 0.5 * dt {
                s.v = s.v.add(burn.dv);
            }
        }
        let a0 = acceleration(s.r, config, t);
        s.v = s.v.add(a0.scale(0.5 * dt));
        s.r = s.r.add(s.v.scale(dt));
        let a1 = acceleration(s.r, config, t + dt);
        s.v = s.v.add(a1.scale(0.5 * dt));
        t += dt;
        out.push(s);
    }
    out
}

// ---------------------------------------------------------------------------
// Adaptive Dormand-Prince (RK45 / DOPRI5) integrator.
// ---------------------------------------------------------------------------

// Butcher tableau for the Dormand-Prince 5(4) pair. The 5th-order weights
// coincide with the final stage row (the method is FSAL), so the seventh stage
// state is reused as the step output.

// Stage nodes c_i.
const DP_C2: f64 = 1.0 / 5.0;
const DP_C3: f64 = 3.0 / 10.0;
const DP_C4: f64 = 4.0 / 5.0;
const DP_C5: f64 = 8.0 / 9.0;
const DP_C6: f64 = 1.0;

// Stage coupling coefficients a_ij (a_72, a_77 are zero).
const DP_A21: f64 = 1.0 / 5.0;

const DP_A31: f64 = 3.0 / 40.0;
const DP_A32: f64 = 9.0 / 40.0;

const DP_A41: f64 = 44.0 / 45.0;
const DP_A42: f64 = -56.0 / 15.0;
const DP_A43: f64 = 32.0 / 9.0;

const DP_A51: f64 = 19372.0 / 6561.0;
const DP_A52: f64 = -25360.0 / 2187.0;
const DP_A53: f64 = 64448.0 / 6561.0;
const DP_A54: f64 = -212.0 / 729.0;

const DP_A61: f64 = 9017.0 / 3168.0;
const DP_A62: f64 = -355.0 / 33.0;
const DP_A63: f64 = 46732.0 / 5247.0;
const DP_A64: f64 = 49.0 / 176.0;
const DP_A65: f64 = -5103.0 / 18656.0;

// Final row, which doubles as the 5th-order solution weights b_i.
const DP_A71: f64 = 35.0 / 384.0;
const DP_A73: f64 = 500.0 / 1113.0;
const DP_A74: f64 = 125.0 / 192.0;
const DP_A75: f64 = -2187.0 / 6784.0;
const DP_A76: f64 = 11.0 / 84.0;

// 4th-order solution weights b*_i (b*_2 is zero).
const DP_B4_1: f64 = 5179.0 / 57600.0;
const DP_B4_3: f64 = 7571.0 / 16695.0;
const DP_B4_4: f64 = 393.0 / 640.0;
const DP_B4_5: f64 = -92097.0 / 339200.0;
const DP_B4_6: f64 = 187.0 / 2100.0;
const DP_B4_7: f64 = 1.0 / 40.0;

/// Normalised (dimensionless) relative error estimate for one DP step.
///
/// Position and velocity error contributions are each scaled by the larger of
/// the old/new magnitude and combined in an L2 sense. This makes the value
/// directly comparable to a relative tolerance regardless of the units of the
/// state.
fn relative_error(old: &State, new: &State, dr: Vec3, dv: Vec3) -> f64 {
    let scale_r = old.r.norm().max(new.r.norm()).max(f64::MIN_POSITIVE);
    let scale_v = old.v.norm().max(new.v.norm()).max(f64::MIN_POSITIVE);
    let er = dr.norm() / scale_r;
    let ev = dv.norm() / scale_v;
    ((er * er + ev * ev) / 2.0).sqrt()
}

/// Advise a next step size from a normalised error and target tolerance.
///
/// `err` is the dimensionless estimate returned by [`dormand_prince_step`] and
/// `rtol` is the tolerance stored on the controller. The classic elementary
/// controller `h_new = h * safety * (rtol / err)^(1/5)` is used, clamped to the
/// configured growth/shrink limits.
fn recommend_dt(
    dt: f64,
    err: f64,
    rtol: f64,
    safety: f64,
    min_factor: f64,
    max_factor: f64,
) -> f64 {
    if err.is_finite() && err > 0.0 {
        let factor = safety * (rtol / err).powf(0.2);
        dt * factor.clamp(min_factor, max_factor)
    } else {
        dt * max_factor
    }
}

/// Perform one adaptive Dormand-Prince 5(4) step of size `dt`.
///
/// `accel` returns the acceleration at a position and time; the state
/// derivative is `(r', v') = (v, a(r, t))`. Returns the 5th-order solution and
/// a dimensionless relative-error estimate suitable for step-size control.
pub fn dormand_prince_step<F>(state: State, accel: &F, t: f64, dt: f64) -> (State, f64)
where
    F: Fn(Vec3, f64) -> Vec3,
{
    // Stage 1 (FSAL: k1 is the derivative at the step start).
    let k1r = state.v;
    let k1v = accel(state.r, t);

    // Stage 2.
    let r2 = state.r.add(k1r.scale(dt * DP_A21));
    let v2 = state.v.add(k1v.scale(dt * DP_A21));
    let k2r = v2;
    let k2v = accel(r2, t + DP_C2 * dt);

    // Stage 3.
    let r3 = state
        .r
        .add(k1r.scale(dt * DP_A31))
        .add(k2r.scale(dt * DP_A32));
    let v3 = state
        .v
        .add(k1v.scale(dt * DP_A31))
        .add(k2v.scale(dt * DP_A32));
    let k3r = v3;
    let k3v = accel(r3, t + DP_C3 * dt);

    // Stage 4.
    let r4 = state
        .r
        .add(k1r.scale(dt * DP_A41))
        .add(k2r.scale(dt * DP_A42))
        .add(k3r.scale(dt * DP_A43));
    let v4 = state
        .v
        .add(k1v.scale(dt * DP_A41))
        .add(k2v.scale(dt * DP_A42))
        .add(k3v.scale(dt * DP_A43));
    let k4r = v4;
    let k4v = accel(r4, t + DP_C4 * dt);

    // Stage 5.
    let r5 = state
        .r
        .add(k1r.scale(dt * DP_A51))
        .add(k2r.scale(dt * DP_A52))
        .add(k3r.scale(dt * DP_A53))
        .add(k4r.scale(dt * DP_A54));
    let v5 = state
        .v
        .add(k1v.scale(dt * DP_A51))
        .add(k2v.scale(dt * DP_A52))
        .add(k3v.scale(dt * DP_A53))
        .add(k4v.scale(dt * DP_A54));
    let k5r = v5;
    let k5v = accel(r5, t + DP_C5 * dt);

    // Stage 6.
    let r6 = state
        .r
        .add(k1r.scale(dt * DP_A61))
        .add(k2r.scale(dt * DP_A62))
        .add(k3r.scale(dt * DP_A63))
        .add(k4r.scale(dt * DP_A64))
        .add(k5r.scale(dt * DP_A65));
    let v6 = state
        .v
        .add(k1v.scale(dt * DP_A61))
        .add(k2v.scale(dt * DP_A62))
        .add(k3v.scale(dt * DP_A63))
        .add(k4v.scale(dt * DP_A64))
        .add(k5v.scale(dt * DP_A65));
    let k6r = v6;
    let k6v = accel(r6, t + DP_C6 * dt);

    // 5th-order solution (uses the final tableau row as its weights).
    let new_r = state
        .r
        .add(k1r.scale(dt * DP_A71))
        .add(k3r.scale(dt * DP_A73))
        .add(k4r.scale(dt * DP_A74))
        .add(k5r.scale(dt * DP_A75))
        .add(k6r.scale(dt * DP_A76));
    let new_v = state
        .v
        .add(k1v.scale(dt * DP_A71))
        .add(k3v.scale(dt * DP_A73))
        .add(k4v.scale(dt * DP_A74))
        .add(k5v.scale(dt * DP_A75))
        .add(k6v.scale(dt * DP_A76));

    // Stage 7 evaluated at the 5th-order solution (FSAL reuse).
    let k7r = new_v;
    let k7v = accel(new_r, t + dt);

    // 4th-order solution for the embedded error estimate.
    let r4s = state
        .r
        .add(k1r.scale(dt * DP_B4_1))
        .add(k3r.scale(dt * DP_B4_3))
        .add(k4r.scale(dt * DP_B4_4))
        .add(k5r.scale(dt * DP_B4_5))
        .add(k6r.scale(dt * DP_B4_6))
        .add(k7r.scale(dt * DP_B4_7));
    let v4s = state
        .v
        .add(k1v.scale(dt * DP_B4_1))
        .add(k3v.scale(dt * DP_B4_3))
        .add(k4v.scale(dt * DP_B4_4))
        .add(k5v.scale(dt * DP_B4_5))
        .add(k6v.scale(dt * DP_B4_6))
        .add(k7v.scale(dt * DP_B4_7));

    let new_state = State { r: new_r, v: new_v };
    let err = relative_error(&state, &new_state, new_r.sub(r4s), new_v.sub(v4s));
    (new_state, err)
}

/// Adaptive step-size controller for the Dormand-Prince integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveController {
    /// Target relative tolerance (normalised error threshold).
    pub rtol: f64,
    /// Safety factor applied to the raw step-size recommendation.
    pub safety: f64,
    /// Minimum allowed step-size shrink factor per step.
    pub min_factor: f64,
    /// Maximum allowed step-size growth factor per step.
    pub max_factor: f64,
    /// Absolute lower bound on the step size (seconds).
    pub dt_min: f64,
    /// Absolute upper bound on the step size (seconds).
    pub dt_max: f64,
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self {
            rtol: 1e-9,
            safety: 0.9,
            min_factor: 0.2,
            max_factor: 5.0,
            dt_min: 1e-6,
            dt_max: 1e6,
        }
    }
}

impl AdaptiveController {
    /// Build a controller for a given relative tolerance with sensible defaults.
    pub fn with_tolerance(rtol: f64) -> Self {
        Self {
            rtol,
            ..Self::default()
        }
    }

    /// Is the error estimate acceptable at this tolerance?
    pub fn accept(&self, err: f64) -> bool {
        err <= self.rtol
    }

    /// Recommend the next step size, clamped to `[dt_min, dt_max]`.
    pub fn next_dt(&self, dt: f64, err: f64) -> f64 {
        recommend_dt(
            dt,
            err,
            self.rtol,
            self.safety,
            self.min_factor,
            self.max_factor,
        )
        .clamp(self.dt_min, self.dt_max)
    }
}

/// Result of an adaptive propagation, including the sampled times and step
/// statistics (useful for diagnostics and tests).
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveTrajectory {
    /// Sample times, starting at the configured epoch.
    pub t: Vec<f64>,
    /// States at each sample time (`states.len() == t.len()`).
    pub states: Vec<State>,
    /// Number of steps that satisfied the tolerance.
    pub accepted: usize,
    /// Number of steps that were rejected and retried.
    pub rejected: usize,
}

impl AdaptiveTrajectory {
    /// Final state of the propagation (the initial state if nothing was stepped).
    pub fn last(&self) -> &State {
        self.states.last().expect("trajectory is never empty")
    }
}

/// Propagate `state` from `config.epoch` to `t_end` using adaptive
/// Dormand-Prince steps with the supplied controller.
///
/// Impulsive burns from the configuration are honoured: the step size is
/// clamped so a step never crosses a burn epoch, and the burn Δv is applied
/// exactly at that epoch.
pub fn propagate_adaptive_with_controller(
    state: State,
    t_end: f64,
    dt0: f64,
    controller: &AdaptiveController,
    config: &Config,
) -> AdaptiveTrajectory {
    let accel = |r: Vec3, t: f64| acceleration(r, config, t);

    let mut s = state;
    let mut t = config.epoch;
    let mut dt = dt0.clamp(controller.dt_min, controller.dt_max);

    let mut traj = AdaptiveTrajectory {
        t: vec![t],
        states: vec![s],
        accepted: 0,
        rejected: 0,
    };

    let forward = t_end >= t;
    if !forward && t_end == t {
        return traj;
    }

    // Sort burn epochs that lie strictly inside the span of integration.
    let mut burn_times: Vec<f64> = config
        .burns
        .iter()
        .map(|b| b.t)
        .filter(|bt| {
            if forward {
                *bt > t && *bt < t_end
            } else {
                *bt < t && *bt > t_end
            }
        })
        .collect();
    burn_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut next_burn = 0usize;

    // Apply any burns scheduled exactly at the start epoch.
    for b in &config.burns {
        if (b.t - t).abs() <= 1e-9 * dt.abs().max(1.0) {
            s.v = s.v.add(b.dv);
        }
    }
    if let Some(first) = traj.states.first_mut() {
        *first = s;
    }

    let tol_eps = 1e-12 * t_end.abs().max(1.0);
    let mut guard = 0usize;
    while (forward && t < t_end && (t_end - t).abs() > tol_eps)
        || (!forward && t > t_end && (t - t_end).abs() > tol_eps)
    {
        guard += 1;
        if guard > 50_000_000 {
            break;
        }

        let remaining = (t_end - t).abs();
        let mut h = dt.min(remaining);

        // Never overshoot the next scheduled burn.
        if forward {
            if let Some(&nb) = burn_times.get(next_burn) {
                if nb > t && nb - t < h {
                    h = nb - t;
                }
            }
        }

        if h <= 0.0 || !h.is_finite() {
            break;
        }

        let (candidate, err) = dormand_prince_step(s, &accel, t, h);

        let finite = candidate.r.x.is_finite()
            && candidate.r.y.is_finite()
            && candidate.r.z.is_finite()
            && candidate.v.x.is_finite()
            && candidate.v.y.is_finite()
            && candidate.v.z.is_finite();

        if !finite {
            // Numerical breakdown: shrink and retry, or give up.
            if h <= controller.dt_min {
                break;
            }
            dt = (h * 0.5).clamp(controller.dt_min, controller.dt_max);
            traj.rejected += 1;
            continue;
        }

        let hit_min = h <= controller.dt_min;
        if controller.accept(err) || hit_min {
            s = candidate;
            t += h;
            if (t_end - t).abs() <= tol_eps {
                t = t_end;
            }
            traj.accepted += 1;
            dt = controller.next_dt(h, err);
            traj.t.push(t);
            traj.states.push(s);

            // Apply a burn if this step landed exactly on its epoch.
            if forward {
                while let Some(&nb) = burn_times.get(next_burn) {
                    if t >= nb - tol_eps {
                        for b in &config.burns {
                            if (b.t - nb).abs() <= tol_eps {
                                s.v = s.v.add(b.dv);
                            }
                        }
                        if let Some(last) = traj.states.last_mut() {
                            *last = s;
                        }
                        next_burn += 1;
                    } else {
                        break;
                    }
                }
            }
        } else {
            traj.rejected += 1;
            let mut next = controller.next_dt(h, err);
            if !next.is_finite() || next >= h {
                next = h * 0.5;
            }
            dt = next.clamp(controller.dt_min, controller.dt_max);
        }
    }

    traj
}

/// Propagate `state` over `[config.epoch, t_end]` with adaptive Dormand-Prince
/// steps targeting the relative tolerance `rtol`.
pub fn propagate_adaptive_detailed(
    state: State,
    t_end: f64,
    dt0: f64,
    rtol: f64,
    config: &Config,
) -> AdaptiveTrajectory {
    let controller = AdaptiveController::with_tolerance(rtol);
    propagate_adaptive_with_controller(state, t_end, dt0, &controller, config)
}

/// Convenience wrapper matching [`propagate`]: adaptively propagate over the
/// span `[config.epoch, t_end]` and return the sampled states only.
pub fn propagate_adaptive(
    state: State,
    t_end: f64,
    dt0: f64,
    rtol: f64,
    config: &Config,
) -> Vec<State> {
    propagate_adaptive_detailed(state, t_end, dt0, rtol, config).states
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_body_config() -> Config {
        Config {
            moon_distance: 0.0,
            mu_moon: 0.0,
            ..Config::default()
        }
    }

    #[test]
    fn two_body_energy_conservation() {
        let cfg = two_body_config();
        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        let steps = 200;
        let dt = 60.0;
        let traj = propagate(state, steps, dt, &cfg);
        let energy = |st: &State| -> f64 { 0.5 * st.v.dot(st.v) - cfg.mu_earth / st.r.norm() };
        let e0 = energy(&state);
        let e1 = energy(traj.last().unwrap());
        let rel = (e1 - e0).abs() / e0.abs();
        assert!(rel < 1e-6, "relative energy drift {rel}");
    }

    #[test]
    fn circular_orbit_closure() {
        let cfg = two_body_config();
        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        // One orbital period.
        let period = 2.0 * PI * (r0.powi(3) / cfg.mu_earth).sqrt();
        let dt = 60.0;
        let steps = (period / dt).round() as usize;
        let traj = propagate(state, steps, dt, &cfg);
        let last = traj.last().unwrap();
        let pos_err = last.r.sub(state.r).norm();
        assert!(pos_err < 5e5, "position error {pos_err}");
    }

    #[test]
    fn earth_moon_finite_step() {
        let cfg = Config::default();
        let state = State {
            r: Vec3::new(6.6e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let traj = propagate(state, 5, 60.0, &cfg);
        for s in &traj {
            assert!(s.r.x.is_finite() && s.r.y.is_finite() && s.r.z.is_finite());
            assert!(s.v.x.is_finite() && s.v.y.is_finite() && s.v.z.is_finite());
        }
    }

    fn circular_two_body(r0: f64) -> (Config, State, f64) {
        let cfg = two_body_config();
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        let period = 2.0 * PI * (r0.powi(3) / cfg.mu_earth).sqrt();
        (cfg, state, period)
    }

    fn specific_energy(st: &State, mu: f64) -> f64 {
        0.5 * st.v.dot(st.v) - mu / st.r.norm()
    }

    #[test]
    fn dp_step_error_is_fifth_order() {
        let (cfg, state, _) = circular_two_body(7.0e6);
        let accel = |r: Vec3, t: f64| acceleration(r, &cfg, t);
        let h = 600.0;
        let (_, err_full) = dormand_prince_step(state, &accel, 0.0, h);
        let (_, err_half) = dormand_prince_step(state, &accel, 0.0, h * 0.5);
        assert!(err_full > 0.0 && err_half > 0.0);
        let ratio = err_full / err_half;
        // Error estimate is O(dt^5): halving dt should cut it by ~2^5 = 32.
        assert!(
            ratio > 16.0 && ratio < 64.0,
            "unexpected convergence ratio {ratio}"
        );
    }

    #[test]
    fn dp_step_tiny_dt_is_identity() {
        let (cfg, state, _) = circular_two_body(7.0e6);
        let accel = |r: Vec3, t: f64| acceleration(r, &cfg, t);
        let (next, err) = dormand_prince_step(state, &accel, 0.0, 1e-9);
        assert!(err < 1e-12, "error estimate {err}");
        assert!(next.r.sub(state.r).norm() < 1e-3);
        assert!(next.v.sub(state.v).norm() < 1e-6);
    }

    #[test]
    fn adaptive_energy_conservation_two_body() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let e0 = specific_energy(&state, cfg.mu_earth);
        let traj = propagate_adaptive_detailed(state, period, 60.0, 1e-10, &cfg);
        let e1 = specific_energy(traj.last(), cfg.mu_earth);
        let rel = (e1 - e0).abs() / e0.abs();
        assert!(rel < 1e-9, "adaptive relative energy drift {rel}");
        assert!(traj.accepted > 0, "no adaptive steps were taken");
        assert!(traj.t.len() == traj.states.len());
    }

    #[test]
    fn adaptive_closure_over_one_period() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let traj = propagate_adaptive_detailed(state, period, 60.0, 1e-10, &cfg);
        assert_eq!(traj.t.last().copied(), Some(period));
        let pos_err = traj.last().r.sub(state.r).norm();
        assert!(pos_err < 1.0e3, "adaptive position error {pos_err}");
    }

    #[test]
    fn adaptive_beats_fixed_step_verlet() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let e0 = specific_energy(&state, cfg.mu_earth);

        // Fixed-step velocity Verlet over the same span.
        let dt = 60.0;
        let steps = (period / dt).round() as usize;
        let verlet = propagate(state, steps, dt, &cfg);
        let verlet_last = verlet.last().unwrap();
        let verlet_energy_drift =
            (specific_energy(verlet_last, cfg.mu_earth) - e0).abs() / e0.abs();
        let verlet_pos_err = verlet_last.r.sub(state.r).norm();

        // Adaptive Dormand-Prince over the same span.
        let traj = propagate_adaptive_detailed(state, period, dt, 1e-11, &cfg);
        let dp_last = traj.last();
        let dp_energy_drift = (specific_energy(dp_last, cfg.mu_earth) - e0).abs() / e0.abs();
        let dp_pos_err = dp_last.r.sub(state.r).norm();

        assert!(
            dp_energy_drift < verlet_energy_drift,
            "DP energy drift {dp_energy_drift} !< Verlet {verlet_energy_drift}"
        );
        assert!(
            dp_pos_err < verlet_pos_err,
            "DP position error {dp_pos_err} !< Verlet {verlet_pos_err}"
        );
    }

    #[test]
    fn adaptive_accepts_looser_tolerance_with_larger_steps() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let loose = propagate_adaptive_detailed(state, period, 60.0, 1e-6, &cfg);
        let tight = propagate_adaptive_detailed(state, period, 60.0, 1e-11, &cfg);
        assert!(
            loose.accepted < tight.accepted,
            "looser tolerance should need fewer steps ({} vs {})",
            loose.accepted,
            tight.accepted
        );
    }

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(a.sub(b), Vec3::new(-3.0, -3.0, -3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.dot(b), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
        assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
    }
}

#[cfg(test)]
mod burn_tests {
    use super::*;

    #[test]
    fn impulsive_burn_changes_velocity() {
        let cfg = Config::default();
        let state = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let mut cfg_burn = cfg.clone();
        cfg_burn.burns.push(Burn {
            t: 0.0,
            dv: Vec3::new(0.0, 10.0, 0.0),
        });
        let traj_none = propagate(state, 2, 60.0, &cfg);
        let traj_burn = propagate(state, 2, 60.0, &cfg_burn);
        // After the burn, velocity should differ.
        let v_none = traj_none.last().unwrap().v.y;
        let v_burn = traj_burn.last().unwrap().v.y;
        assert!(
            (v_burn - v_none).abs() > 5.0,
            "burn did not affect velocity"
        );
    }
}
