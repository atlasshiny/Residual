//! Minimal Rust core for the Earth-Moon restricted dynamics.
//!
//! TODO: adaptive Dormand-Prince integrator and full SPICE ephemeris are not
//! implemented yet.

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

    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

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
    let z2_over_r2 = (d.z * d.z) / r2;
    let k = -1.5 * config.j2 * config.mu_earth * config.earth_radius * config.earth_radius
        / (r2 * r2 * r);
    let xy_factor = 1.0 - 5.0 * z2_over_r2;
    let z_factor = 3.0 - 5.0 * z2_over_r2;
    Vec3::new(k * xy_factor * d.x, k * xy_factor * d.y, k * z_factor * d.z)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn two_body_config() -> Config {
        let mut c = Config::default();
        c.moon_distance = 0.0;
        c.mu_moon = 0.0;
        c
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
        let state = State { r: Vec3::new(7.0e6, 0.0, 0.0), v: Vec3::new(0.0, 7.5e3, 0.0) };
        let mut cfg_burn = cfg.clone();
        cfg_burn.burns.push(Burn { t: 60.0, dv: Vec3::new(0.0, 10.0, 0.0) });
        let traj_none = propagate(state, 2, 60.0, &cfg);
        let traj_burn = propagate(state, 2, 60.0, &cfg_burn);
        // After the burn, velocity should differ.
        let v_none = traj_none.last().unwrap().v.y;
        let v_burn = traj_burn.last().unwrap().v.y;
        assert!((v_burn - v_none).abs() > 5.0, "burn did not affect velocity");
    }
}
