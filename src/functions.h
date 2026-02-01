#pragma once
#include <cstddef>

float cosine_distance_avx(const float* const a, const float* const b, const size_t size);

float cosine_distance(const float* const a, const float* const b, const size_t size);

