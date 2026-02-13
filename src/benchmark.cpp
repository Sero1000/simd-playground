#include <algorithm>
#include <cmath>
#include <benchmark/benchmark.h>
#include <random>

#include "functions.h"
#include "test_inputs.h"
// Benchmark
static void BM_cosine_distance(benchmark::State& state)
{
    for (auto _ : state)
    {
	double result;
        // Prevent compiler from optimizing away the result
	cosine_distance_avx(a.data(), b.data(), a.size(), &result);
	benchmark::DoNotOptimize(result);
    }
}

static void BM_clamp_basic(benchmark::State& state)
{
    constexpr size_t SIZE = 1024 * 1024;
    // Create RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate source and destination buffers
    std::vector<float> src(SIZE);
    std::vector<float> dst(SIZE);

    // Fill source with random data
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(gen);
    }

    for(auto _ : state)
    {
	clamp_basic(src.data(), src.size(), 0.25, 0.75, dst.data());

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
}

static void BM_clamp_avx(benchmark::State& state)
{
    constexpr size_t SIZE = 1024 * 1024;
    // Create RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate source and destination buffers
    std::vector<float> src(SIZE);
    std::vector<float> dst(SIZE);

    // Fill source with random data
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(gen);
    }

    for(auto _ : state)
    {
	clamp_avx(src.data(), src.size(), 0.25, 0.75, dst.data());

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
}

static void BM_predicate_count(benchmark::State& state)
{
    constexpr size_t SIZE = 1024 * 1024;
    // Create RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate source and destination buffers
    std::vector<float> src(SIZE);
    std::vector<float> dst(SIZE);

    // Fill source with random data
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(gen);
    }

    size_t result;
    for(auto _ : state)
    {
	count_predicate(src.data(), 0.5, src.size(), &result);

	benchmark::DoNotOptimize(&result);
	benchmark::ClobberMemory();
    }
}

static void BM_predicate_count_avx(benchmark::State& state)
{
    constexpr size_t SIZE = 1024 * 1024;
    // Create RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate source and destination buffers
    std::vector<float> src(SIZE);

    // Fill source with random data
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(gen);
    }

    size_t result;
    for(auto _ : state)
    {
	count_predicate_avx(src.data(), 0.5, src.size(), &result);

	benchmark::DoNotOptimize(&result);
	benchmark::ClobberMemory();
    }
}

static void BM_find_min_basic(benchmark::State& state)
{
    constexpr size_t SIZE = 1024 * 1024;
    // Create RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate source and destination buffers
    std::vector<float> src(SIZE);

    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(gen);
    }

    float min;
    for(auto _ : state)
    {
	auto min = std::min_element(src.begin(),src.end());
	benchmark::DoNotOptimize(&min);
	benchmark::ClobberMemory();
    }
}

static void BM_find_min_avx(benchmark::State& state)
{
    constexpr size_t SIZE = 1024 * 1024;
    // Create RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate source and destination buffers
    std::vector<float> src(SIZE);

    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(gen);
    }

    float min;
    for(auto _ : state)
    {
	find_min_avx(src.data(), src.size(), &min);

	benchmark::DoNotOptimize(&min);
	benchmark::ClobberMemory();
    }
}

static void BM_separate(benchmark::State& state)
{
    constexpr size_t ELEMENTS = 1024 * 1024;
    constexpr size_t BUFFER_SIZE = ELEMENTS * 3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 10);

    std::vector<float> elements(BUFFER_SIZE);
    for(size_t i = 0; i < BUFFER_SIZE; ++i)
    {
	elements[i] = dist(gen);
    }

    std::vector<float> x_buffer(ELEMENTS), y_buffer(ELEMENTS), z_buffer(ELEMENTS);

    for(auto _ : state)
    {
	separate_basic(elements.data(), elements.size(), x_buffer.data(), y_buffer.data(), z_buffer.data());

	benchmark::DoNotOptimize(x_buffer.data());
	benchmark::DoNotOptimize(y_buffer.data());
	benchmark::DoNotOptimize(z_buffer.data());
	benchmark::ClobberMemory();
    }
}

static void BM_separate_avx(benchmark::State& state)
{
    constexpr size_t ELEMENTS = 1024 * 1024;
    constexpr size_t BUFFER_SIZE = ELEMENTS * 3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 10);

    std::vector<float> elements(BUFFER_SIZE);
    for(size_t i = 0; i < BUFFER_SIZE; ++i)
    {
	elements[i] = dist(gen);
    }

    std::vector<float> x_buffer(ELEMENTS), y_buffer(ELEMENTS), z_buffer(ELEMENTS);

    for(auto _ : state)
    {
	separate_avx(elements.data(), elements.size(), x_buffer.data(), y_buffer.data(), z_buffer.data());

	benchmark::DoNotOptimize(x_buffer.data());
	benchmark::DoNotOptimize(y_buffer.data());
	benchmark::DoNotOptimize(z_buffer.data());
	benchmark::ClobberMemory();
    }
}

static void BM_transpose_avx(benchmark::State& state)
{
    constexpr size_t rows = 8;
    constexpr size_t columns = 8;

    float matrix [rows * columns];
    float transposed_matrix[rows * columns];


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 10);

    for(int i = 0; i < rows * columns; ++i)
    {
	matrix [i] = dist(gen);
    }

    for(auto _ : state)
    {
	transpose_avx(matrix, rows, columns, transposed_matrix);

	benchmark::DoNotOptimize(transposed_matrix);
	benchmark::ClobberMemory();
    }
}

static void BM_transpose_basic(benchmark::State& state)
{
    constexpr size_t rows = 8;
    constexpr size_t columns = 8;

    float matrix [rows * columns];
    float transposed_matrix[rows * columns];


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 10);

    for(int i = 0; i < rows * columns; ++i)
    {
	matrix [i] = dist(gen);
    }

    for(auto _ : state)
    {
	transpose_basic(matrix, rows, columns, transposed_matrix);

	benchmark::DoNotOptimize(transposed_matrix);
	benchmark::ClobberMemory();
    }
}


BENCHMARK(BM_transpose_basic);
BENCHMARK(BM_transpose_avx);
BENCHMARK_MAIN();
