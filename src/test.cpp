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
    constexpr size_t SIZE = 1024 * 1024;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    std::vector<uint8_t> cacheEvict(SIZE);

    {
      std::vector<float> src(SIZE); 
      std::vector<float> dst(SIZE); 

      // Fill source with random data
      for (size_t i = 0; i < SIZE; ++i) {
          src[i] = dist(gen);
      }

      for (size_t i = 0; i < SIZE; ++i) {
	++cacheEvict[i];
      }
      auto begin = std::chrono::high_resolution_clock::now();
      
      clamp_basic(src.data(), src.size(), 0.25, 0.75, dst.data());
      // for (size_t i = 0; i < SIZE; ++i)
      //     dst[i] = src[i];

      auto end = std::chrono::high_resolution_clock::now();

      auto ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      std::cout<<"Basic copy : "<<ms.count()<<" ns."<<std::endl;
    }
    {
      std::vector<float> src(SIZE); 
      std::vector<float> dst(SIZE); 

      // Fill source with random data
      for (size_t i = 0; i < SIZE; ++i) {
          src[i] = dist(gen);
      }

      for (size_t i = 0; i < SIZE; ++i) {
	++cacheEvict[i];
      }

      auto begin = std::chrono::high_resolution_clock::now();
      clamp_avx(src.data(), src.size(), 0.25, 0.75, dst.data());
      auto end = std::chrono::high_resolution_clock::now();

      auto ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      std::cout<<"Basic copy : "<<ms.count()<<" ns."<<std::endl;
    }
}
