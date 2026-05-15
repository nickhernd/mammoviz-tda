#pragma once
#include <vector>
#include <array>
#include <memory>
#include "io/VolumeData.h"
#include "xai/GradCAM.h"

namespace mmviz::render {

// GPU-accelerated volume renderer using ray casting (OpenGL 4.6 compute shaders)
// Supports: MIP, DVR (direct volume rendering), isosurface, GradCAM overlay
class VolumeRenderer {
public:
    enum class RenderMode {
        MaxIntensityProjection,  // MIP — fast overview
        DirectVolumeRendering,   // DVR with transfer function
        Isosurface,              // marching cubes isosurface
        GradCAMOverlay,          // DVR + colored saliency overlay
    };

    struct TransferFunction {
        // Maps intensity [0,1] → RGBA color+opacity
        std::vector<std::array<float,4>> lut;  // 256-entry lookup table
        static TransferFunction breastDefault();
        static TransferFunction calcificationHighlight();
    };

    VolumeRenderer();
    ~VolumeRenderer();

    void uploadVolume(const io::VolumeData& vol);
    void uploadSaliency(const xai::GradCAM::SaliencyMap& map);

    void setRenderMode(RenderMode mode);
    void setTransferFunction(const TransferFunction& tf);
    void setCameraPos(float x, float y, float z);
    void setGradCAM(bool enabled);

    // Called each frame from RenderEngine
    void render(const float* view_matrix, const float* proj_matrix);

    // Returns the set of voxel coordinates corresponding to a persistence point
    // Used for linked selection: click diagram → highlight tissue
    std::vector<std::array<int,3>> voxelsForPersistenceRegion(
        float birth, float death, int dimension) const;

private:
    unsigned int m_vao       = 0;
    unsigned int m_vbo       = 0;
    unsigned int m_ebo       = 0;
    unsigned int m_vol_tex   = 0;
    unsigned int m_sal_tex   = 0;
    unsigned int m_tf_tex    = 0;
    unsigned int m_shader    = 0;
    int          m_num_idx   = 0;
    bool         m_ready     = false;
    std::array<int,3> m_vol_dims = {1,1,1};

    RenderMode     m_mode = RenderMode::GradCAMOverlay;
    TransferFunction m_tf;
    float m_cam_x = 0.5f, m_cam_y = 0.5f, m_cam_z = 2.5f;
    bool  m_gradcam_on = true;

    void lazyInit();
};

} // namespace mmviz::render
