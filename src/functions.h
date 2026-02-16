#pragma once
#include <cstddef>
#include <cstdint>

void cosine_distance_avx(const float* const a, const float* const b, const size_t size, double* result);

float cosine_distance(const float* const a, const float* const b, const size_t size);

void copy_avx(const uint8_t* const src, uint8_t* const dst, const size_t size);

extern "C" void copy_basic(const uint8_t* const src, uint8_t* const dst, const size_t size);

extern "C" void clamp_basic(const float* const src, const size_t size, const float min, const float max, float* const output);

extern "C" void clamp_avx(const float* const src, const size_t size, const float min, const float max, float* const output);

// Counting how many elements are smaller than a given number
extern "C" void count_predicate(const float* const src, const float limit, const size_t size, size_t* result);

extern "C" void count_predicate_avx(const float* const src, const float limit, const size_t size, size_t* result);

extern "C" void find_min(const float* const src, const size_t size, float* const result);

extern "C" void find_min_avx(const float* const src, const size_t size, float* const result);

// Seperates an array of 3D point coordinates into separate arrays
extern "C" void separate_basic(const float* const src, const size_t size, float* const x_result, float* const y_result, float* const z_result);

extern "C" void separate_avx(const float* const src, const size_t size, float* const x_result, float* const y_result, float* const z_result);

extern "C" void transpose_basic(const float* const src, const size_t row_size, const size_t column_size, float* const transposed_matrix);

extern "C" void transpose_sse(const float* const src, const size_t row_size, const size_t column_size, float* const transposed_matrix);

extern "C" void transpose_avx(const float* const src, const size_t row_size, const size_t column_size, float* const transposed_matrix);

extern "C" void reference_for_branchless_computation(const float* const src, const size_t size, float* const result);

extern "C" void branchless_computation_avx(const float* const src, const size_t size, float* const result);

extern "C" void count_numbers(const char* const src, const size_t size, size_t* const result);

extern "C" void count_numbers_avx(const char* const src, const size_t size, size_t* const result);
