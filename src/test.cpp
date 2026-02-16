#include "functions.h"
#include "test_inputs.h"

#include <algorithm>
#include <cctype>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>

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

TEST(Count, CountTest)
{
    size_t result;
    count_predicate(golden_compare_input_data.data(), golden_compare_limit, golden_compare_input_data.size(), &result);

    EXPECT_EQ(result, golden_compare_result);
}

TEST(Count, CountTestAVX)
{
    size_t result;
    count_predicate_avx(golden_compare_input_data.data(), golden_compare_limit, golden_compare_input_data.size(), &result);

    EXPECT_EQ(result, golden_compare_result);
}

TEST(Min, MinTestAVX)
{
    auto stl_min = std::min_element(golden_compare_input_data.begin(), golden_compare_input_data.end());
    float avx_result;
    find_min_avx(golden_compare_input_data.data(), golden_compare_input_data.size(), &avx_result);

    EXPECT_EQ(*stl_min, avx_result);
}

TEST(Separate, SeparateTest)
{
    constexpr size_t ELEMENTS = 1024 * 1024;
    constexpr size_t BUFFER_SIZE = ELEMENTS * 3;

    std::vector<float> elements(BUFFER_SIZE);
    for(size_t i = 0; i < ELEMENTS; ++i)
    {
	elements[3 * i] = i;
	elements[3 * i + 1] = i;
	elements[3 * i + 2] = i;
    }

    std::vector<float> x_buffer(ELEMENTS), y_buffer(ELEMENTS), z_buffer(ELEMENTS);

    separate_basic(elements.data(), elements.size(), x_buffer.data(), y_buffer.data(), z_buffer.data());

    for(size_t i = 0; i < ELEMENTS;++i)
    {
	EXPECT_EQ(x_buffer[i], elements[3 * i])<<", index : "<<i;
	EXPECT_EQ(y_buffer[i], elements[3 * i + 1])<<", index : "<<i;
	EXPECT_EQ(z_buffer[i], elements[3 * i + 2])<<", index : "<<i;
    }
}

TEST(Separate, SeparateTestAVX)
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

    separate_avx(elements.data(), elements.size(), x_buffer.data(), y_buffer.data(), z_buffer.data());

    for(size_t i = 0; i < ELEMENTS;++i)
    {
	EXPECT_EQ(x_buffer[i], elements[3 * i])<<", index : "<<i;
	EXPECT_EQ(y_buffer[i], elements[3 * i + 1])<<", index : "<<i;
	EXPECT_EQ(z_buffer[i], elements[3 * i + 2])<<", index : "<<i;
    }
}

TEST(Transpose, TransposeTestBasic)
{
    constexpr size_t rows = 8;
    constexpr size_t columns = 8;

    float matrix [rows * columns];
    float transposed_matrix[rows * columns];

    for(int i = 0; i < rows * columns; ++i)
    {
	matrix [i] = i + 1;
    }

    transpose_basic(matrix, rows, columns, transposed_matrix);

    for(size_t row = 0; row < rows; ++row)
    {
	std::cout<<"[ ";
	for(size_t column = 0; column < columns; ++column)
	{
	    std::cout<<transposed_matrix[row * columns + column]<<" ";
	}
	std::cout<<" ]"<<std::endl;
    }
    int pix = 4;
    ++pix;
}

TEST(Transpose, TransposeTestAVX)
{
    constexpr size_t rows = 8;
    constexpr size_t columns = 8;

    float matrix [rows * columns];
    float transposed_matrix[rows * columns];

    for(int i = 0; i < rows * columns; ++i)
    {
	matrix [i] = i + 1;
    }

    transpose_avx(matrix, rows, columns, transposed_matrix);

    for(size_t row = 0; row < rows; ++row)
    {
	std::cout<<"[ ";
	for(size_t column = 0; column < columns; ++column)
	{
	    std::cout<<transposed_matrix[row * columns + column]<<" ";
	}
	std::cout<<" ]"<<std::endl;
    }
}

TEST(Branchless, BranchlessTestAvx)
{
    constexpr size_t SIZE = 8;
    std::vector<float> input = {1,2,-1,-2,3,4,-3,-4};

    constexpr float sqrt_2 = std::sqrt(2.0f);
    constexpr float sqrt_3 = std::sqrt(3.0f);

    std::vector<float> expected = {1, sqrt_2, 1, 4, sqrt_3, 2, 9, 16};

    std::vector<float> result(SIZE);

    branchless_computation_avx(input.data(), input.size(), result.data());
    for(size_t i = 0; i < SIZE; ++i)
    {
	EXPECT_FLOAT_EQ(expected[i], result[i])<<", index : "<<i;
    }
}

TEST(Count, AVX)
{
    const size_t text_size = 1024;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    std::string test_input;
    test_input.reserve(text_size);

    for(int i = 0; i < text_size; ++i)
    {
	test_input.push_back(dist(gen));
    }

    size_t avxResult;

    size_t number_count = std::count_if(test_input.begin(), test_input.end(), [](char el){return std::isdigit(el);});

    count_numbers_avx(test_input.data(), test_input.size(), &avxResult);

    EXPECT_EQ(avxResult, number_count);
}
