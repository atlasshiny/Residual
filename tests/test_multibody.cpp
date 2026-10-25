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
    EXPECT_LT(rel, 5e-6);  // Verlet energy oscillates ~4e-6; phase-dependent
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

TEST(Multibody, EnsembleStatsMatchesReference) {
#ifdef SKIP_CUDA_TESTS
    GTEST_SKIP() << "Skipping CUDA test in CI";
#endif
    const int bodies = 137;
    std::vector<MbsState> states(bodies);
    for (int i = 0; i < bodies; ++i) {
        states[i].r = {static_cast<double>(i + 1), static_cast<double>(i * 2), static_cast<double>(i % 5)};
        states[i].v = {0.5 * i, -0.25 * i, 1.0};
    }

    MbsEnsembleStats stats{};
    const int rc = mbs_ensemble_stats(states.data(), bodies, &stats);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(stats.count, bodies);

    double sx = 0, sy = 0, sz = 0, svx = 0, svy = 0, svz = 0, sr2 = 0, sv2 = 0;
    double min_r2 = std::numeric_limits<double>::infinity();
    double max_r2 = -std::numeric_limits<double>::infinity();
    for (const auto& state : states) {
        sx += state.r.x;
        sy += state.r.y;
        sz += state.r.z;
        svx += state.v.x;
        svy += state.v.y;
        svz += state.v.z;
        const double r2 = state.r.x * state.r.x + state.r.y * state.r.y + state.r.z * state.r.z;
        const double v2 = state.v.x * state.v.x + state.v.y * state.v.y + state.v.z * state.v.z;
        sr2 += r2;
        sv2 += v2;
        min_r2 = std::min(min_r2, r2);
        max_r2 = std::max(max_r2, r2);
    }
    EXPECT_NEAR(stats.sum_x, sx, 1e-6);
    EXPECT_NEAR(stats.sum_y, sy, 1e-6);
    EXPECT_NEAR(stats.sum_z, sz, 1e-6);
    EXPECT_NEAR(stats.sum_vx, svx, 1e-6);
    EXPECT_NEAR(stats.sum_vy, svy, 1e-6);
    EXPECT_NEAR(stats.sum_vz, svz, 1e-6);
    EXPECT_NEAR(stats.sum_r2, sr2, 1e-6);
    EXPECT_NEAR(stats.sum_v2, sv2, 1e-6);
    EXPECT_NEAR(stats.min_r2, min_r2, 1e-6);
    EXPECT_NEAR(stats.max_r2, max_r2, 1e-6);
}

TEST(Multibody, EnsembleStatsDeviceResident) {
#ifdef SKIP_CUDA_TESTS
    GTEST_SKIP() << "Skipping CUDA test in CI";
#endif
    const int bodies = 257;
    std::vector<MbsState> host(bodies);
    for (int i = 0; i < bodies; ++i) {
        host[i].r = {static_cast<double>(i), static_cast<double>(i + 1), 0.0};
        host[i].v = {1.0, 2.0, 3.0};
    }

    MbsState* device_states = nullptr;
    MbsEnsembleStats* device_stats = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_states), bodies * sizeof(MbsState)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_stats), sizeof(MbsEnsembleStats)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_states, host.data(), bodies * sizeof(MbsState), cudaMemcpyHostToDevice), cudaSuccess);

    const int rc = mbs_ensemble_stats_device(device_states, bodies, device_stats);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    MbsEnsembleStats stats{};
    ASSERT_EQ(cudaMemcpy(&stats, device_stats, sizeof(MbsEnsembleStats), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(stats.count, bodies);
    EXPECT_GT(stats.sum_r2, 0.0);

    cudaFree(device_stats);
    cudaFree(device_states);
}

TEST(Multibody, EnsembleStatsRejectsInvalidInput) {
    MbsState state{};
    MbsEnsembleStats stats{};
    EXPECT_NE(mbs_ensemble_stats(nullptr, 1, &stats), 0);
    EXPECT_NE(mbs_ensemble_stats(&state, 0, &stats), 0);
    EXPECT_NE(mbs_ensemble_stats(&state, 1, nullptr), 0);
}