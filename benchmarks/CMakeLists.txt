if(NOT SKIP_CUDA_TESTS)
    add_executable(multibody_bench multibody_bench.cpp)
    target_link_libraries(multibody_bench PRIVATE multibody_solver benchmark::benchmark)

    # CUDA vs Rust comparison.  The Rust side is built as a static library by
    # cargo and linked into backend_bench through a narrow C ABI.
    find_program(CARGO_EXECUTABLE cargo)
    if(NOT CARGO_EXECUTABLE AND EXISTS "$ENV{HOME}/.cargo/bin/cargo")
        set(CARGO_EXECUTABLE "$ENV{HOME}/.cargo/bin/cargo")
    endif()

    if(CARGO_EXECUTABLE)
        find_package(Threads REQUIRED)

        set(RUST_BENCH_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/rust-target")
        set(RUST_BENCH_LIB
            "${RUST_BENCH_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}dynamics_bench${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(RUST_BENCH_STAMP "${CMAKE_CURRENT_BINARY_DIR}/dynamics_bench_rust.stamp")

        add_custom_command(
            OUTPUT "${RUST_BENCH_STAMP}"
            BYPRODUCTS "${RUST_BENCH_LIB}"
            COMMAND "${CARGO_EXECUTABLE}" build --release -p dynamics-bench
                    --target-dir "${RUST_BENCH_TARGET_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E touch "${RUST_BENCH_STAMP}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            DEPENDS
                "${CMAKE_SOURCE_DIR}/Cargo.toml"
                "${CMAKE_SOURCE_DIR}/crates/dynamics/Cargo.toml"
                "${CMAKE_SOURCE_DIR}/crates/dynamics/src/lib.rs"
                "${CMAKE_SOURCE_DIR}/crates/dynamics-bench/Cargo.toml"
                "${CMAKE_SOURCE_DIR}/crates/dynamics-bench/src/lib.rs"
            COMMENT "Building Rust dynamics-bench static library"
            VERBATIM
        )

        add_custom_target(dynamics_bench_rust DEPENDS "${RUST_BENCH_STAMP}")

        add_library(dynamics_bench STATIC IMPORTED GLOBAL)
        set_target_properties(dynamics_bench PROPERTIES
            IMPORTED_LOCATION "${RUST_BENCH_LIB}")

        add_executable(backend_bench backend_bench.cpp)
        add_dependencies(backend_bench dynamics_bench_rust)
        target_link_libraries(backend_bench PRIVATE
            multibody_solver benchmark::benchmark dynamics_bench Threads::Threads)
        if(UNIX)
            target_link_libraries(backend_bench PRIVATE ${CMAKE_DL_LIBS} m)
        endif()
    else()
        message(WARNING
            "cargo not found; backend_bench (CUDA vs Rust) will not be built")
    endif()
endif()
