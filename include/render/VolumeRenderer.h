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
        std::vector<std::array<float,4>> lut;
        static TransferFunction breastDefault();
        static TransferFunction calcificationHighlight();
    };

    // Per-layer rendering parameters (controlled from ImGui)
    struct LayerParams {
        struct Layer {
            float r = 1.f, g = 1.f, b = 1.f;
            float opacity = 1.f;
            bool  visible = true;
        };
        Layer fat     = {0.95f, 0.82f, 0.35f, 1.f, true};  // warm yellow
        Layer gland   = {0.85f, 0.45f, 0.55f, 1.f, true};  // rose pink
        Layer dense   = {1.00f, 0.78f, 0.55f, 1.f, true};  // orange
        Layer calc    = {1.00f, 0.97f, 0.65f, 1.f, true};  // bright yellow-white
        float lesion_sal_thresh = 0.08f;
        float lesion_opacity    = 1.00f;
        bool  lesion_visible    = true;
        float clip_x = 1.f, clip_y = 1.f, clip_z = 1.f;   // 1=no clip
    };

    VolumeRenderer();
    ~VolumeRenderer();

    void uploadVolume(const io::VolumeData& vol);
    void uploadSaliency(const xai::GradCAM::SaliencyMap& map);

    void setRenderMode(RenderMode mode);
    void setTransferFunction(const TransferFunction& tf);
    void setLayerParams(const LayerParams& lp);
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
    LayerParams    m_layers;
    float m_cam_x = 0.5f, m_cam_y = 0.5f, m_cam_z = 2.5f;
    bool  m_gradcam_on = true;

    void lazyInit();
};

} // namespace mmviz::render
