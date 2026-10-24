// Shared definitions for the CUDA multibody solver.
//
// Equation of motion for each test particle:
//
//   r'' = -mu_E (r - r_E) / |r - r_E|^3
//         -mu_M (r - r_M) / |r - r_M|^3
//         + a_perturbation
//
// The Earth and Moon follow a circular barycentric ephemeris in the xy plane.
// a_perturbation currently supports Earth's J2 oblateness term (0 = off).
#pragma once

#include "macros.cuh"

#include <cmath>

// Public data layout. These are plain C-compatible structs so callers can
// mirror them without including this header.

// Cartesian vector. Positions are metres; velocities are m/s.
struct MbsVec3 {
    double x;
    double y;
    double z;
};

// Position and velocity of one test particle.
struct MbsState {
    MbsVec3 r;  // position (m)
    MbsVec3 v;  // velocity (m/s)
};

// Simulation parameters. Fill this with mbs_default_config() and override
// only the fields that need to change.
struct MbsConfig {
    double dt;              // integration step, seconds
    double epoch;           // time of the initial states, seconds
    double mu_earth;        // Earth gravitational parameter, m^3/s^2
    double mu_moon;         // Moon gravitational parameter, m^3/s^2
    double earth_radius;    // equatorial radius, metres
    double j2;              // Earth J2 coefficient, dimensionless
    double moon_distance;   // Earth-Moon separation, metres
    double moon_period;     // 0 pins the Moon in place
    double moon_phase;      // radians at t = 0
    double softening;       // metres; 0 gives the exact point-mass terms (default)
    int perturbation;       // 0 = none, 1 = Earth J2
};

namespace mbs {

// Physical constants in SI units.
constexpr double kPi = 3.141592653589793238462643383279502884;

constexpr double kMuEarth = 3.986004418e14;       // m^3/s^2
constexpr double kMuMoon = 4.9048695e12;          // m^3/s^2
constexpr double kEarthRadius = 6.378137e6;       // m
constexpr double kEarthJ2 = 1.08262668e-3;        // dimensionless
constexpr double kEarthMoonDistance = 3.844e8;    // m
constexpr double kMoonPeriod = 2.3605915e6;       // s (27.321661 days)

// Small vector helpers used by the device physics below.
MBS_DEVICE_INLINE MbsVec3 operator+(const MbsVec3& a, const MbsVec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

MBS_DEVICE_INLINE MbsVec3 operator-(const MbsVec3& a, const MbsVec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

MBS_DEVICE_INLINE MbsVec3 operator*(const MbsVec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

MBS_DEVICE_INLINE MbsVec3& operator+=(MbsVec3& a, const MbsVec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

MBS_DEVICE_INLINE double dot(const MbsVec3& a, const MbsVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Circular barycentric ephemeris. Both bodies orbit the origin with the
// Moon's angular rate; positions are in the inertial xy plane. Weighting by
// gravitational parameters instead of masses keeps the barycentre exact.
MBS_DEVICE_INLINE void earth_moon_positions(double t, const MbsConfig& config, MbsVec3& earth, MbsVec3& moon) {
    if (config.moon_distance <= 0.0) {
        earth = {};
        moon = {};
        return;
    }

    double theta = config.moon_phase;
    const double total_mu = config.mu_earth + config.mu_moon;
    // Circular motion is Keplerian: omega^2 = total_mu / D^3.
    // moon_period is retained for API compatibility but is not used here.
    if (config.moon_distance > 0.0 && total_mu > 0.0) {
        theta += sqrt(total_mu / (config.moon_distance *
                                  config.moon_distance *
                                  config.moon_distance)) * t;
    }

    const double moon_fraction =
        (total_mu > 0.0) ? (config.mu_moon / total_mu) : 0.0;

    const double c = cos(theta);
    const double s = sin(theta);

    earth = {-config.moon_distance * moon_fraction * c,
             -config.moon_distance * moon_fraction * s, 0.0};
    moon = {config.moon_distance * (1.0 - moon_fraction) * c,
            config.moon_distance * (1.0 - moon_fraction) * s, 0.0};
}

// J2 acceleration relative to the Earth's centre (d = r - r_E).
// z is the Earth spin axis; the ephemeris plane is equatorial in this model.
MBS_DEVICE_INLINE MbsVec3 earth_j2_acceleration(const MbsVec3& d, const MbsConfig& config) {
    const double r2 = dot(d, d);
    if (r2 <= 0.0) {
        return {};
    }

    const double r = sqrt(r2);
    if (r <= config.earth_radius) {
        return {};  // J2 is only valid outside the body.
    }
    const double z2_over_r2 = (d.z * d.z) / r2;

    // -3/2 * J2 * mu * R^2 / r^5
    const double k = -1.5 * config.j2 * config.mu_earth * config.earth_radius * config.earth_radius / (r2 * r2 * r);
    const double xy_factor = 1.0 - 5.0 * z2_over_r2;
    const double z_factor = 3.0 - 5.0 * z2_over_r2;

    return {k * xy_factor * d.x, k * xy_factor * d.y, k * z_factor * d.z};
}

// Sum of the Earth and Moon point-mass terms plus the optional perturbation.
// Softening replaces |d|^3 with (|d|^2 + eps^2)^(3/2) to keep close
// encounters finite.
MBS_DEVICE_INLINE MbsVec3 acceleration(const MbsVec3& r, const MbsConfig& config, double t) {
    MbsVec3 earth{};
    MbsVec3 moon{};
    earth_moon_positions(t, config, earth, moon);

    const MbsVec3 d_earth = r - earth;
    const MbsVec3 d_moon = r - moon;

    const double eps2 = config.softening * config.softening;
    const double r_earth2 = dot(d_earth, d_earth) + eps2;
    const double r_moon2 = dot(d_moon, d_moon) + eps2;

    MbsVec3 a{};
    if (config.mu_earth > 0.0) {
        a += d_earth * (-config.mu_earth / (r_earth2 * sqrt(r_earth2)));
    }
    if (config.mu_moon > 0.0 && config.moon_distance > 0.0) {
        a += d_moon * (-config.mu_moon / (r_moon2 * sqrt(r_moon2)));
    }
    if (config.perturbation == 1) {
        a += earth_j2_acceleration(d_earth, config);
    }

    return a;
}

// One kick-drift-kick velocity-Verlet step: half kick, drift, re-evaluate
// acceleration, second half kick. Symplectic for the conservative terms.
MBS_DEVICE_INLINE void velocity_verlet_step(MbsState& state, double t, double dt, const MbsConfig& config) {
    const MbsVec3 a0 = acceleration(state.r, config, t);
    state.v += a0 * (0.5 * dt);
    state.r += state.v * dt;
    const MbsVec3 a1 = acceleration(state.r, config, t + dt);
    state.v += a1 * (0.5 * dt);
}

}  // namespace mbs

extern "C" {

// Fills config with the SI defaults above and disables perturbations.
void mbs_default_config(MbsConfig* config);

// Propagates num_bodies independent particles for num_steps steps.
//
// trajectory may be null. When provided, it must hold
// (num_steps + 1) * num_bodies states, laid out as
//   trajectory[step * num_bodies + body]
// with the initial state at step 0. final_states may be null when a
// trajectory is requested.
//
// The call is synchronous. Returns cudaSuccess (0) on success, otherwise a
// cudaError_t value suitable for cudaGetErrorString().
// All positions are barycentric: Earth is offset from the origin by about
// 4,673 km, so convert geocentric initial states before calling.
int mbs_propagate(const MbsState* initial_states, int num_bodies, int num_steps, const MbsConfig* config, MbsState* final_states, MbsState* trajectory);

// Device-resident variant: propagates in place on memory the caller already
// owns.  device_states and device_trajectory must be device pointers; the
// trajectory buffer follows the same layout as mbs_propagate and may be null.
//
// This launches the kernel and returns without synchronising, so callers can
// time pure GPU work with CUDA events and reuse allocations across calls.
// Validate the launch result before reading the buffers.
int mbs_propagate_device(MbsState* device_states, int num_bodies, int num_steps, const MbsConfig* config, MbsState* device_trajectory);

}  // extern "C"
