#include "functions.h"
#include "test_inputs.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

// Helper to compare floats with tolerance
constexpr float kEpsilon = 1e-5f;

TEST(CosineDistanceAVXTest, AVXUnit) {
    double result;
    cosine_distance_avx(a.data(), b.data(), a.size(), &result);
    float aaa = cosine_distance(a.data(), b.data(), a.size());

    // Cosine similarity = 1.0 → distance = 0.0
    EXPECT_NEAR(result, golden_cosine_distance, kEpsilon);
}

TEST(CosineDistanceAVXTest, AVXSimple) {
    constexpr size_t array_size = 8;
    std::vector<float> a_test(array_size), b_test(array_size);

    for(int i = 1; i <= array_size; ++i)
    {
	a_test[i - 1] = i;
	b_test[i - 1] = i;
    }

    double result;
    cosine_distance_avx(a_test.data(), b_test.data(), a_test.size(), &result);

    // Cosine similarity = 1.0 → distance = 0.0
    EXPECT_NEAR(result, 0, kEpsilon);
}

TEST(Clamp, ClampTest)
{
    std::vector<float> result(golden_clamped_input.size());
    clamp_avx(golden_clamped_input.data(), golden_clamped_input.size(), golden_clamped_min, golden_clamped_max, result.data());

    for(size_t i = 0; i < golden_clamped_input.size(); ++i)
    {
	EXPECT_NEAR(golden_clamped[i], result[i], kEpsilon) << "index="<<i;
    }
}
