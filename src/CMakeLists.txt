add_library(mammoviz_utils  utils/Logger.cpp utils/Config.cpp utils/Timer.cpp)
add_library(mammoviz_io     io/DicomLoader.cpp io/VolumeData.cpp)
add_library(mammoviz_tda_lib tda/PointCloud.cpp tda/VietorisRips.cpp tda/PersistenceDiagram.cpp)
add_library(mammoviz_nn     nn/ModelInference.cpp nn/FeatureExtractor.cpp)
add_library(mammoviz_xai    xai/GradCAM.cpp xai/ManifoldProjector.cpp)
add_library(mammoviz_render render/RenderEngine.cpp render/VolumeRenderer.cpp
                             render/DiagramRenderer.cpp render/ManifoldRenderer.cpp)
add_library(mammoviz_ui     ui/MainWindow.cpp ui/VolumePanel.cpp
                             ui/DiagramPanel.cpp ui/ManifoldPanel.cpp)

set(COMMON_INCLUDES ${CMAKE_SOURCE_DIR}/include)
set(GUDHI_INCLUDES ${CMAKE_BINARY_DIR}/_deps/gudhi-src/include)

# ── utils (no external deps) ──────────────────────────────────────────────────
target_include_directories(mammoviz_utils PUBLIC ${COMMON_INCLUDES})

# ── io (depends on ITK) ───────────────────────────────────────────────────────
target_include_directories(mammoviz_io PUBLIC ${COMMON_INCLUDES})
target_link_libraries(mammoviz_io PUBLIC mammoviz_utils ${ITK_LIBRARIES})

# ── tda (depends on GUDHI + Eigen3) ──────────────────────────────────────────
target_include_directories(mammoviz_tda_lib PUBLIC ${COMMON_INCLUDES})
target_link_libraries(mammoviz_tda_lib PUBLIC mammoviz_utils mammoviz_io gudhi Eigen3::Eigen)

# ── nn (depends on ONNX Runtime) ─────────────────────────────────────────────
target_include_directories(mammoviz_nn PUBLIC ${COMMON_INCLUDES})
target_link_libraries(mammoviz_nn PUBLIC mammoviz_utils mammoviz_io)
if(TARGET onnxruntime)
    target_link_libraries(mammoviz_nn PUBLIC onnxruntime)
endif()

# ── xai (depends on Eigen3 + nn) ─────────────────────────────────────────────
target_include_directories(mammoviz_xai PUBLIC ${COMMON_INCLUDES})
target_link_libraries(mammoviz_xai PUBLIC mammoviz_utils mammoviz_nn Eigen3::Eigen)

# ── render (depends on OpenGL + GLFW + GLEW) ─────────────────────────────────
target_include_directories(mammoviz_render PUBLIC ${COMMON_INCLUDES})
target_link_libraries(mammoviz_render PUBLIC
    mammoviz_utils mammoviz_io mammoviz_tda_lib mammoviz_xai
    OpenGL::GL glfw GLEW::GLEW)

# ── ui ────────────────────────────────────────────────────────────────────────
target_include_directories(mammoviz_ui PUBLIC ${COMMON_INCLUDES})
target_link_libraries(mammoviz_ui PUBLIC mammoviz_render)
if(TARGET imgui)
    target_link_libraries(mammoviz_ui PUBLIC imgui)
endif()

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(mammoviz main.cpp)
target_include_directories(mammoviz PRIVATE ${COMMON_INCLUDES})
target_link_libraries(mammoviz PRIVATE
    mammoviz_io mammoviz_tda_lib mammoviz_nn mammoviz_xai
    mammoviz_render mammoviz_ui mammoviz_utils)
