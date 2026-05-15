include(GoogleTest)

add_executable(test_tda unit/test_tda.cpp)
target_link_libraries(test_tda PRIVATE
    mammoviz_tda
    mammoviz_io
    mammoviz_utils
    GTest::GTest
    GTest::Main
)
target_include_directories(test_tda PRIVATE ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(test_tda)
