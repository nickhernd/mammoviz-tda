add_library(mammoviz_io     io/DicomLoader.cpp io/VolumeData.cpp)
add_library(mammoviz_tda    tda/PointCloud.cpp tda/VietorisRips.cpp tda/PersistenceDiagram.cpp)
add_library(mammoviz_nn     nn/ModelInference.cpp nn/FeatureExtractor.cpp)
add_library(mammoviz_xai    xai/GradCAM.cpp xai/ManifoldProjector.cpp)
add_library(mammoviz_render render/RenderEngine.cpp render/VolumeRenderer.cpp render/DiagramRenderer.cpp render/ManifoldRenderer.cpp)
add_library(mammoviz_ui     ui/MainWindow.cpp ui/VolumePanel.cpp ui/DiagramPanel.cpp ui/ManifoldPanel.cpp)
add_library(mammoviz_utils  utils/Logger.cpp utils/Config.cpp utils/Timer.cpp)

set(ALL_MODULES mammoviz_io mammoviz_tda mammoviz_nn mammoviz_xai mammoviz_render mammoviz_ui mammoviz_utils)

foreach(mod ${ALL_MODULES})
  target_include_directories(${mod} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${GUDHI_INCLUDE_DIRS}
  )
  target_link_libraries(${mod} PUBLIC
    Eigen3::Eigen
    ${ITK_LIBRARIES}
  )
endforeach()

target_link_libraries(mammoviz_render PUBLIC OpenGL::GL glfw)
target_link_libraries(mammoviz_ui     PUBLIC OpenGL::GL glfw)

if(TARGET onnxruntime::onnxruntime)
  target_link_libraries(mammoviz_nn PUBLIC onnxruntime::onnxruntime)
endif()

# Main executable
add_executable(mammoviz main.cpp)
target_link_libraries(mammoviz PRIVATE ${ALL_MODULES})
