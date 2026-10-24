#include "multibody_solver.cuh"

#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#ifdef SKIP_CUDA_TESTS
#define CUDA_TEST
#else
#define CUDA_TEST
#endif

static bool finite3(const MbsVec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

TEST(Multibody, TwoBodyOrbitConservesEnergy) {
#ifdef SKIP_CUDA_TESTS
    GTEST_SKIP() << "Skipping CUDA test in CI";
#endif
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
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(finite3(final_state.r));
    ASSERT_TRUE(finite3(final_state.v));

    const double r_final =
        std::sqrt(final_state.r.x * final_state.r.x +
                  final_state.r.y * final_state.r.y +
                  final_state.r.z * final_state.r.z);
    EXPECT_LT(std::fabs(r_final - r0), 10.0);

    auto energy = [&](const MbsState& s) {
        const double rr = std::sqrt(s.r.x * s.r.x + s.r.y * s.r.y + s.r.z * s.r.z);
        const double ke = 0.5 * (s.v.x * s.v.x + s.v.y * s.v.y + s.v.z * s.v.z);
        return ke - cfg.mu_earth / rr;
    };
    const double e0 = energy(initial);
    const double e1 = energy(final_state);
    const double rel = std::fabs(e1 - e0) / std::fabs(e0);
    EXPECT_LT(rel, 1e-7);
}

TEST(Multibody, EarthMoonStepStaysFinite) {
#ifdef SKIP_CUDA_TESTS
    GTEST_SKIP() << "Skipping CUDA test in CI";
#endif
    MbsConfig cfg;
    mbs_default_config(&cfg);
    cfg.dt = 60.0;
    MbsState initial{{6.6e6, 0.0, 0.0}, {0.0, 7.5e3, 0.0}};
    MbsState final_state{};
    const int rc = mbs_propagate(&initial, 1, 1, &cfg, &final_state, nullptr);
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(finite3(final_state.r));
    ASSERT_TRUE(finite3(final_state.v));
}

TEST(Multibody, InvalidConfigRejected) {
    MbsConfig cfg;
    mbs_default_config(&cfg);
    cfg.dt = 0.0;
    MbsState initial{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    MbsState final_state{};
    const int rc = mbs_propagate(&initial, 1, 1, &cfg, &final_state, nullptr);
    EXPECT_NE(rc, 0);
}