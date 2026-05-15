#pragma once
#include <memory>
#include <functional>

struct GLFWwindow;

namespace mmviz::render {

class VolumeRenderer;
class DiagramRenderer;
class ManifoldRenderer;

struct RenderConfig {
    int   width  = 1920;
    int   height = 1080;
    bool  vsync  = true;
    bool  msaa4x = true;
    float fov    = 45.0f;
};

// Orchestrates the multi-panel interactive visualization
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

    using ClickCallback = std::function<void(float birth, float death, int dim)>;
    void onDiagramClick(ClickCallback cb);

    bool isRunning() const;

private:
    RenderConfig m_cfg;
    GLFWwindow*  m_window = nullptr;

    std::shared_ptr<VolumeRenderer>   m_volume;
    std::shared_ptr<DiagramRenderer>  m_diagram;
    std::shared_ptr<ManifoldRenderer> m_manifold;

    ClickCallback m_click_cb;

    void processInput();
    void renderFrame();
};

} // namespace mmviz::render
