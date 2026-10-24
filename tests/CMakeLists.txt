add_executable(multibody_test test_multibody.cpp)
target_link_libraries(multibody_test PRIVATE multibody_solver GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(multibody_test)