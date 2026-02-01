#include "functions.h"

#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <cmath>

__attribute__((always_inline, target("avx2,fma"))) inline static float approximate(float magnitude_a, float magnitude_b, float dot_product)
{
    __m128 magnitudes = _mm_set_ps(magnitude_a, magnitude_b, magnitude_a, magnitude_b);
    __m128 rsqrt = _mm_rsqrt_ps(magnitudes);

    __m128 half = _mm_set_ps(0.5, 0.5, 0.5, 0.5);
    __m128 one_and_half = _mm_set_ps(1.5, 1.5, 1.5, 1.5);

    __m128 newton_result = _mm_mul_ps(rsqrt, _mm_sub_ps(one_and_half, _mm_mul_ps(half, _mm_mul_ps(magnitudes, _mm_mul_ps(rsqrt, rsqrt)))));

    float a_reciprocal = _mm_cvtss_f32(newton_result);
    float b_reciprocal = _mm_cvtss_f32(_mm_shuffle_ps(newton_result, newton_result, _MM_SHUFFLE2(1, 1)));

    return a_reciprocal * b_reciprocal * dot_product;
}
__attribute__((always_inline, target("avx2,fma"))) inline static float reduce(__m256 vec)
{
    __m128 lo = _mm256_castps256_ps128(vec);
    __m128 hi = _mm256_extractf128_ps(vec, 1);
    __m128 sum = _mm_add_ps(lo, hi);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
float cosine_distance_avx(const float* const a, const float* const b, const size_t size)
{
    __m256 dot_product = _mm256_setzero_ps();
    __m256 magnitude_a = _mm256_setzero_ps();
    __m256 magnitude_b = _mm256_setzero_ps();

    for(size_t i = 0; i < size; i+=8)
    {
	__m256 ai = _mm256_loadu_ps(a + i);
	__m256 bi = _mm256_loadu_ps(b + i);

	magnitude_a = _mm256_fmadd_ps(ai, ai, magnitude_a);
	magnitude_b = _mm256_fmadd_ps(bi, bi, magnitude_b);
	dot_product = _mm256_fmadd_ps(ai, bi, dot_product);
    }

    float dot_product_sum = reduce(dot_product);
    float magnitude_a_sum = reduce(magnitude_a);
    float magnitude_b_sum = reduce(magnitude_b);

    return 1 - approximate(magnitude_a_sum, magnitude_b_sum, dot_product_sum);
}

float cosine_distance(const float* const a, const float* const b, const size_t size)
{
  float dot_product = 0;
  float magnitude_a = 0;
  float magnitude_b = 0;

  for(size_t i = 0; i < size; ++i)
  {
    float a_value = a[i];
    float b_value = b[i];

    magnitude_a += a_value * a_value;
    magnitude_b += b_value * b_value;
    dot_product += a_value * b_value;
  }

  float rsqrt_a = 1 / std::sqrt(magnitude_a);
  float rsqrt_b = 1 / std::sqrt(magnitude_b);

  return 1 - (dot_product / ( rsqrt_a * rsqrt_b));
}

