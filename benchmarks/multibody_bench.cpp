#include "multibody_solver.cuh"

#include <benchmark/benchmark.h>

static void BM_PropagateOneStep(benchmark::State& state) {
    MbsConfig cfg;
    mbs_default_config(&cfg);
    cfg.dt = 60.0;
    cfg.moon_distance = 0.0;
    MbsState initial{{7.0e6, 0.0, 0.0}, {0.0, 7.5e3, 0.0}};
    for (auto _ : state) {
        MbsState out{};
        mbs_propagate(&initial, 1, 1, &cfg, &out, nullptr);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_PropagateOneStep);

BENCHMARK_MAIN();