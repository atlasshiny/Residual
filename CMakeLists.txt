cmake_minimum_required(VERSION 3.24)

project(multibody_solver LANGUAGES CXX CUDA)

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

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src/kernels)

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

install(TARGETS multibody_solver
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(FILES src/kernels/multibody_solver.cuh
        DESTINATION include)