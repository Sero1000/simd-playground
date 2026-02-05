cmake_minimum_required(VERSION 4.0)

project(playground)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(benchmark CONFIG REQUIRED)

enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(benchmark)
target_sources(benchmark PRIVATE src/benchmark.cpp src/functions.cpp)
target_link_libraries(benchmark PRIVATE benchmark::benchmark benchmark::benchmark_main)
target_compile_options(benchmark PRIVATE -O3 -fopt-info-vec-optimized)

add_executable(unit_test)
target_sources(unit_test PRIVATE src/test.cpp src/functions.cpp)
target_link_libraries(unit_test PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

add_test(AllTestsInMain unit_test)
