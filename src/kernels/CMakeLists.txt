add_library(multibody_solver STATIC multibody_solver.cu)
target_include_directories(multibody_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multibody_solver PUBLIC CUDA::cudart)

set_target_properties(multibody_solver PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)