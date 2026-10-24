cmake_minimum_required(VERSION 3.24)

project(multibody_solver LANGUAGES CXX)

if(SKIP_CUDA_TESTS)
    # Disable CUDA language for the project if we are skipping CUDA tests in CI
    # This prevents CMake from failing when nvcc is not found.
else()
    # Re-enable CUDA if not skipping
    enable_language(CUDA)
endif()

include(FetchContent)

# GoogleTest for the C++/CUDA test suite.
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark for future CUDA kernel benchmarking.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
FetchContent_MakeAvailable(benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to common consumer/compute architectures when the caller does not
# override CMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

if(NOT SKIP_CUDA_TESTS)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(src/kernels)
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()

# Optional host-side demo. Keep it out of the default build when src/main.cpp
# is empty so an empty translation unit is not compiled.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
   AND NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
   AND NOT "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp" STREQUAL "")
    file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp" _main_size)
    if(_main_size GREATER 0)
        add_executable(multibody_demo src/main.cpp)
        target_link_libraries(multibody_demo PRIVATE multibody_solver)
    endif()
endif()

if(NOT SKIP_CUDA_TESTS)
    install(TARGETS multibody_solver
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

if(NOT SKIP_CUDA_TESTS)
    install(FILES src/kernels/multibody_solver.cuh
            DESTINATION include)
endif()