include(GoogleTest)

# GTest on Arch Linux sets INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "/usr/include"
# on GTest::gtest/gtest_main. CMake emits this as -isystem /usr/include which
# breaks GCC 16's #include_next chain in <cstdlib>. NO_SYSTEM_FROM_IMPORTED
# suppresses auto-promotion of INTERFACE_INCLUDE_DIRECTORIES but does NOT touch
# INTERFACE_SYSTEM_INCLUDE_DIRECTORIES — that must be cleared explicitly.
foreach(_gt_tgt IN ITEMS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    if(TARGET ${_gt_tgt})
        set_target_properties(${_gt_tgt} PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ""
            INTERFACE_INCLUDE_DIRECTORIES        "")
    endif()
endforeach()

add_executable(test_tda unit/test_tda.cpp)
target_link_libraries(test_tda PRIVATE
    mammoviz_tda_lib
    mammoviz_io
    mammoviz_utils
    GTest::GTest
    GTest::Main
)
target_include_directories(test_tda PRIVATE ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(test_tda)
