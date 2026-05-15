#pragma once
#include <memory>
#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace mmviz::render {

class VolumeRenderer;
class DiagramRenderer;
class ManifoldRenderer;

struct RenderConfig {
    int   width  = 1280;
    int   height = 720;
    bool  vsync  = true;
    bool  msaa4x = false;  // off by default for compatibility
    float fov    = 45.0f;
};

// All computed results the UI needs to display
struct AppState {
    // Prediction
    int         pred_class      = -1;   // 0=benign, 1=malignant, -1=unknown
    float       confidence      = 0.f;
    float       logit_benign    = 0.f;
    float       logit_malignant = 0.f;

    // TDA statistics
    int   tda_h0 = 0, tda_h1 = 0, tda_h2 = 0;
    float tda_max_persistence = 0.f;
    int   tda_total_points    = 0;

    // Volume metadata
    int   vol_x = 0, vol_y = 0, vol_z = 0;
    float spacing_xy = 0.f, spacing_z = 0.f;

    // Case info
    std::string case_id;
    std::string pattern;  // "malignant" | "benign" | ""
};

class RenderEngine {
public:
    explicit RenderEngine(const RenderConfig& cfg = RenderConfig{});
    ~RenderEngine();

    void init();
    void run();
    void shutdown();

    void setVolumeRenderer  (std::shared_ptr<VolumeRenderer>   r);
    void setDiagramRenderer (std::shared_ptr<DiagramRenderer>  r);
    void setManifoldRenderer(std::shared_ptr<ManifoldRenderer> r);
    void setAppState        (const AppState& state);

    using ClickCallback = std::function<void(float birth, float death, int dim)>;
    void onDiagramClick(ClickCallback cb);

    bool isRunning() const;

private:
    RenderConfig m_cfg;
    GLFWwindow*  m_window = nullptr;

    std::shared_ptr<VolumeRenderer>   m_volume;
    std::shared_ptr<DiagramRenderer>  m_diagram;
    std::shared_ptr<ManifoldRenderer> m_manifold;

    AppState     m_state;
    ClickCallback m_click_cb;

    // Camera orbit state
    float m_cam_azimuth   = 0.f;
    float m_cam_elevation = 20.f;
    float m_cam_distance  = 2.5f;
    bool  m_dragging      = false;
    double m_last_mx = 0, m_last_my = 0;

    // UI toggles
    bool m_show_gradcam   = true;
    bool m_show_tda_panel = true;
    int  m_tf_mode        = 0;  // 0=breast, 1=calcification highlight

    void processInput();
    void renderFrame();
    void renderImGui();
    void initImGui();
    void shutdownImGui();

    static void mouseButtonCb(GLFWwindow*, int, int, int);
    static void cursorPosCb (GLFWwindow*, double, double);
    static void scrollCb    (GLFWwindow*, double, double);
};

} // namespace mmviz::render
