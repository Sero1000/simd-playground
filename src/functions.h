#pragma once
#include <cstddef>
#include <cstdint>

void cosine_distance_avx(const float* const a, const float* const b, const size_t size, double* result);

float cosine_distance(const float* const a, const float* const b, const size_t size);

void copy_avx(const uint8_t* const src, uint8_t* const dst, const size_t size);

extern "C" void copy_basic(const uint8_t* const src, uint8_t* const dst, const size_t size);

extern "C" void clamp_basic(const float* const src, const size_t size, const float min, const float max, float* const output);

extern "C" void clamp_avx(const float* const src, const size_t size, const float min, const float max, float* const output);
