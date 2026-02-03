#include <cmath>
#include <benchmark/benchmark.h>
#include <random>

#include "functions.h"
#include "test_inputs.h"
// Benchmark
static void BM_cosine_distance(benchmark::State& state)
{
    // const size_t size = state.range(0);
    // const size_t size = 1536;

    // // Allocate once (not measured)
    // std::vector<float> a(size), b(size);

    // // Initialize once
    // std::mt19937 rng(123);
    // std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    // for (size_t i = 0; i < size; ++i)
    // {
    //     a[i] = dist(rng);
    //     b[i] = dist(rng);
    // }

    for (auto _ : state)
    {
	double result;
        // Prevent compiler from optimizing away the result
	cosine_distance_avx(a.data(), b.data(), a.size(), &result);
	benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_cosine_distance);
BENCHMARK_MAIN();
