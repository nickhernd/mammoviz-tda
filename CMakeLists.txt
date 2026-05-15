cmake_minimum_required(VERSION 3.20)
project(mammoviz_tda VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address")

# ---- Dependencies ----

# ITK: Medical image I/O and segmentation
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# GUDHI: Topological Data Analysis
find_package(GUDHI REQUIRED)

# Eigen3: Linear algebra
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# OpenGL + GLFW: Rendering
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# ONNX Runtime: CNN inference
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
  message(WARNING "ONNX Runtime not found - NN inference disabled")
  add_compile_definitions(NO_ONNX)
endif()

# Dear ImGui (vendored via third_party)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)

# GTest
find_package(GTest QUIET)

# ---- Subdirectories ----
add_subdirectory(src)
add_subdirectory(shaders)

if(GTest_FOUND)
  enable_testing()
  add_subdirectory(tests)
endif()
