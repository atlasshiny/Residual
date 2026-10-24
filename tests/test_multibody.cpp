#include "multibody_solver.cuh"

#include <cmath>
#include <cstdio>

static bool finite3(const MbsVec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

int main() {
    // Two-body Earth orbit: energy and radius should be nearly conserved.
    {
        MbsConfig cfg;
        mbs_default_config(&cfg);
        cfg.dt = 60.0;
        cfg.perturbation = 0;
        cfg.softening = 0.0;
        cfg.moon_distance = 0.0; // disable Moon

        const double r0 = 7.0e6;
        const double v0 = std::sqrt(cfg.mu_earth / r0);
        MbsState initial{{r0, 0.0, 0.0}, {0.0, v0, 0.0}};
        const double period = 2.0 * M_PI * std::sqrt(r0 * r0 * r0 / cfg.mu_earth);
        const int steps = static_cast<int>(period / cfg.dt);

        MbsState final_state{};
        const int rc = mbs_propagate(&initial, 1, steps, &cfg, &final_state, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "mbs_propagate returned %d\n", rc);
            return 1;
        }

        const double r_final =
            std::sqrt(final_state.r.x * final_state.r.x +
                      final_state.r.y * final_state.r.y +
                      final_state.r.z * final_state.r.z);
        if (!finite3(final_state.r) || !finite3(final_state.v)) return 2;
        if (std::fabs(r_final - r0) > 10.0) {
            std::fprintf(stderr, "orbit radius drifted %.6f m\n", r_final - r0);
            return 3;
        }

        auto energy = [&](const MbsState& s) {
            const double rr = std::sqrt(s.r.x * s.r.x + s.r.y * s.r.y + s.r.z * s.r.z);
            const double ke = 0.5 * (s.v.x * s.v.x + s.v.y * s.v.y + s.v.z * s.v.z);
            return ke - cfg.mu_earth / rr;
        };
        const double e0 = energy(initial);
        const double e1 = energy(final_state);
        const double rel = std::fabs(e1 - e0) / std::fabs(e0);
        if (rel > 1e-7) {
            std::fprintf(stderr, "relative energy drift %.3e\n", rel);
            return 4;
        }
    }

    // Earth-Moon system: one step should stay finite.
    {
        MbsConfig cfg;
        mbs_default_config(&cfg);
        cfg.dt = 60.0;
        MbsState initial{{6.6e6, 0.0, 0.0}, {0.0, 7.5e3, 0.0}};
        MbsState final_state{};
        const int rc = mbs_propagate(&initial, 1, 1, &cfg, &final_state, nullptr);
        if (rc != 0) return 5;
        if (!finite3(final_state.r) || !finite3(final_state.v)) return 6;
    }

    // Invalid configuration must be rejected.
    {
        MbsConfig cfg;
        mbs_default_config(&cfg);
        cfg.dt = 0.0;
        MbsState initial{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        MbsState final_state{};
        const int rc = mbs_propagate(&initial, 1, 1, &cfg, &final_state, nullptr);
        if (rc == 0) return 7;
    }

    std::printf("multibody tests passed\n");
    return 0;
}