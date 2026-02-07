#include "functions.h"

#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <popcntintrin.h>
#include <xmmintrin.h>
#include <cmath>

__attribute__((always_inline, target("avx2,fma")))
inline static double approximate(double magnitude_a, double magnitude_b, double dot_product)
{
    __m128d magnitudes = _mm_set_pd(magnitude_a, magnitude_b);
    __m128d rsqrt = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(magnitudes)));

    __m128d half = _mm_set_pd(0.5, 0.5);
    __m128d one_and_half = _mm_set_pd(1.5, 1.5);

    __m128d newton_result = _mm_mul_pd(rsqrt, _mm_sub_pd(one_and_half, _mm_mul_pd(half, _mm_mul_pd(magnitudes, _mm_mul_pd(rsqrt, rsqrt)))));

    double b_reciprocal =_mm_cvtsd_f64(_mm_unpackhi_pd(newton_result, newton_result));
    double a_reciprocal = _mm_cvtsd_f64(newton_result);

    return a_reciprocal * b_reciprocal * dot_product;
}

__attribute__((always_inline, target("avx2,fma"))) 
inline static double reduce(__m256 vec)
{
    __m128 lo = _mm256_castps256_ps128(vec);
    __m128 hi = _mm256_extractf128_ps(vec, 1);

    __m256d lo_double = _mm256_cvtps_pd(lo);
    __m256d hi_double = _mm256_cvtps_pd(hi);

    __m256d sum = _mm256_add_pd(lo_double, hi_double);

    __m128d lo_double_2 = _mm256_castpd256_pd128(sum);
    __m128d hi_double_2 = _mm256_extractf128_pd(sum, 1);

    __m128d sum_2 = _mm_add_pd(lo_double_2, hi_double_2);

    __m128d final_sum = _mm_hadd_pd(sum_2, sum_2);

    return _mm_cvtsd_f64(final_sum);
}

__attribute__((target("avx2,fma")))
void cosine_distance_avx(const float* const a, const float* const b, const size_t size, double* result)
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

    double dot_product_sum = reduce(dot_product);
    double magnitude_a_sum = reduce(magnitude_a);
    double magnitude_b_sum = reduce(magnitude_b);

    *result = 1 - approximate(magnitude_a_sum, magnitude_b_sum, dot_product_sum);
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

__attribute__((target("avx2,fma"))) 
void copy_avx(const uint8_t* const src, uint8_t* const dst, const size_t size)
{
    for(size_t i = 0; i + 32 <= size; i+= 32)
    {
        __m256i ints = _mm256_loadu_si256((__m256i*)(src + i));
	_mm256_storeu_si256((__m256i*)(dst + i), ints);
    }
}

void copy_basic(const uint8_t* const src, uint8_t* const dst, const size_t size)
{
      for (size_t i = 0; i < size; ++i)
          dst[i] = src[i];
}

void clamp_basic(const float* const src, const size_t size, const float min, const float max, float* const output)
{
    for(size_t i = 0; i < size; ++i)
    {
	output[i] = std::max(min, std::min(max, src[i]));
    }
}

__attribute__((target("avx2"))) 
void clamp_avx(const float* const src, const size_t size, const float min, const float max, float* const output)
{
    __m256 min_vec = _mm256_set1_ps(min);
    __m256 max_vec = _mm256_set1_ps(max);
    size_t i = 0;
    for(; i + 8 <= size; i+=8) 
    {
	__m256 src_vec = _mm256_load_ps(src + i);

	__m256 clamped = _mm256_min_ps(max_vec, _mm256_max_ps(min_vec, src_vec));

	_mm256_storeu_ps(output + i, clamped);

    }
    for(; i < size; ++i)
    {
	output[i] = std::min(max, std::max(min, src[i]));
    }
}

void count_predicate(const float* const src, const float limit, const size_t size, size_t* result)
{
    size_t count = 0;

    for(int i = 0; i < size; ++i)
    {
	if (src[i] < limit)
	{
	    ++count;
	}
    }

    *result = count;
}

__attribute__((target("avx2"))) 
void count_predicate_avx(const float* const src, const float limit, const size_t size, size_t* result)
{
    __m256 limit_vec = _mm256_set1_ps(limit);
    size_t count = 0;
    size_t i = 0;

    for(; i + 8 <= size; i+=8)
    {
	__m256 vec = _mm256_loadu_ps(src + i);
	__m256 cmp = _mm256_cmp_ps(vec, limit_vec, _CMP_LE_OS);

	int cmp_mask = _mm256_movemask_ps(cmp);
	count += _mm_popcnt_u32(cmp_mask);
    }

    for(;i < size; ++i)
    {
	if (src[i] < limit)
	    ++count;
    }

    *result = count;
}

void find_min(const float* const src, const size_t size, float* const result)
{
    float min = src[0];

    for(size_t i = 0; i < size; ++i)
    {
	if(src[i] < min)
	    min = src[i];
    }

    *result = min;
}

__attribute__((target("avx2"))) 
void find_min_avx(const float* const src, const size_t size, float* const result)
{
    float min = src[0];    

    size_t i = 0;
    for(; i + 8 <= size; i += 8)
    {
	__m256 vec = _mm256_loadu_ps(src + i);

	__m128 lo = _mm256_castps256_ps128(vec);
	__m128 hi = _mm256_extractf128_ps(vec, 1);
	__m128 min128 = _mm_min_ps(lo, hi);

	__m128 shuf = _mm_movehdup_ps(min128);
	__m128 mins = _mm_min_ps(shuf, min128);

	 shuf = _mm_movehl_ps(shuf, mins);
	 mins = _mm_min_ss(mins, shuf);

	 float min_in_vec = _mm_cvtss_f32(mins);
	 if(min_in_vec < min)
	     min = min_in_vec;
    }

    for (;i < size; ++i)
    {
	if(src[i] < min)
	    min = src[i];
    }

    *result = min;
}





