cmake_minimum_required(VERSION 3.20)
project(mammoviz_tda VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prevent CMake from converting INTERFACE_INCLUDE_DIRECTORIES of imported targets
# (OpenGL, GLFW, GLEW) into -isystem flags. On Arch Linux these targets report
# /usr/include as their install prefix, and -isystem /usr/include breaks GCC 16's
# #include_next chain in <cstdlib>. Using -I instead has no functional difference
# since /usr/include is already in GCC's default search path.
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

# ── Workaround: CMake 4.3.x was released before GCC 16 (April 2026) ──────────
# CMake doesn't have GCC 16's feature tables, so CMAKE_CXX_COMPILE_FEATURES
# is empty, causing any target_compile_features() call (e.g. from Eigen3/ITK)
# to fail with "No known features for GNU 16.x".
# GCC 16 is a strict superset of GCC 15, so we inject the full feature list.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0")
    set(_gcc16_features
        cxx_std_11 cxx_std_14 cxx_std_17 cxx_std_20 cxx_std_23
        cxx_aggregate_default_initializers cxx_alias_templates
        cxx_alignas cxx_alignof cxx_attribute_deprecated cxx_attributes
        cxx_auto_type cxx_binary_literals cxx_constexpr
        cxx_contextual_conversions cxx_decltype cxx_decltype_auto
        cxx_decltype_incomplete_return_types cxx_default_function_template_args
        cxx_defaulted_functions cxx_defaulted_move_initializers
        cxx_delegating_constructors cxx_deleted_functions cxx_digit_separators
        cxx_enum_forward_declarations cxx_explicit_conversions
        cxx_extended_friend_declarations cxx_extern_templates cxx_final
        cxx_func_identifier cxx_generalized_initializers cxx_generic_lambdas
        cxx_inheriting_constructors cxx_inline_namespaces
        cxx_lambda_init_captures cxx_lambdas cxx_local_type_template_args
        cxx_long_long_type cxx_noexcept cxx_nonstatic_member_init cxx_nullptr
        cxx_override cxx_range_for cxx_raw_string_literals
        cxx_reference_qualified_functions cxx_relaxed_constexpr
        cxx_return_type_deduction cxx_right_angle_brackets cxx_rvalue_references
        cxx_sizeof_member cxx_static_assert cxx_strong_enums
        cxx_template_template_parameters cxx_thread_local
        cxx_trailing_return_types cxx_unicode_literals cxx_uniform_initialization
        cxx_unrestricted_unions cxx_user_literals cxx_variable_templates
        cxx_variadic_macros cxx_variadic_templates)
    set(CMAKE_CXX_COMPILE_FEATURES    "${_gcc16_features}" CACHE STRING "" FORCE)
    set(CMAKE_CXX14_COMPILE_FEATURES  "cxx_std_14"         CACHE STRING "" FORCE)
    set(CMAKE_CXX17_COMPILE_FEATURES  "cxx_std_17"         CACHE STRING "" FORCE)
    set(CMAKE_CXX20_COMPILE_FEATURES  "cxx_std_20"         CACHE STRING "" FORCE)
    set(CMAKE_CXX23_COMPILE_FEATURES  "cxx_std_23"         CACHE STRING "" FORCE)
    message(STATUS "GCC 16 workaround: injected C++ feature tables for CMake 4.3.x")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address")

include(FetchContent)

# ── ITK: Medical image I/O ────────────────────────────────────────────────────
find_package(ITK REQUIRED COMPONENTS
    ITKCommon ITKIOGDCM ITKIOImageBase ITKIOMeta
    ITKImageIntensity ITKStatistics)
include(${ITK_USE_FILE})

# ── GUDHI: Topological Data Analysis (header-only) ───────────────────────────
# GUDHI has a custom build system — we use FetchContent_Populate to only
# download and extract the source tree, then wire up headers manually.
FetchContent_Declare(gudhi
    URL             https://github.com/GUDHI/gudhi-devel/releases/download/tags%2Fgudhi-release-3.10.0/gudhi.3.10.0.tar.gz
    URL_HASH        MD5=00fcb5b873bb806f959178c4bf5d2f9b
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(gudhi)
if(NOT gudhi_POPULATED)
    FetchContent_Populate(gudhi)
endif()

add_library(gudhi INTERFACE)
target_include_directories(gudhi INTERFACE ${gudhi_SOURCE_DIR}/include)
message(STATUS "GUDHI headers at: ${gudhi_SOURCE_DIR}/src")

# ── Eigen3 ────────────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED NO_MODULE)

# ── OpenGL + GLFW + GLEW ─────────────────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(glfw3  3.3 REQUIRED)
find_package(GLEW        REQUIRED)


# ── ONNX Runtime (pre-built binary in third_party/) ───────────────────────────
set(ONNX_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
if(EXISTS "${ONNX_DIR}/include/onnxruntime_cxx_api.h")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${ONNX_DIR}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNX_DIR}/include")
    message(STATUS "ONNX Runtime found in third_party/")
else()
    message(WARNING "ONNX Runtime not found in third_party/ — NN inference disabled")
    add_compile_definitions(NO_ONNX)
endif()

# ── Dear ImGui (vendored) ─────────────────────────────────────────────────────
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")
if(EXISTS "${IMGUI_DIR}/imgui.h")
    add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
    set(HAVE_IMGUI TRUE)
    message(STATUS "ImGui found — building with GUI overlay")
endif()

# ── GTest ─────────────────────────────────────────────────────────────────────
find_package(GTest QUIET)

# ── Subdirectories ────────────────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(shaders)

if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()
