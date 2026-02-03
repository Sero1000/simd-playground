#pragma once
#include <cstddef>

void cosine_distance_avx(const float* const a, const float* const b, const size_t size, double* result);

float cosine_distance(const float* const a, const float* const b, const size_t size);

