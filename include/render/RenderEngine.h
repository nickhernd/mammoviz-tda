#pragma once
#include <memory>
#include <string>
#include <functional>

struct GLFWwindow;

namespace mmviz::render {

class VolumeRenderer;
class DiagramRenderer;
class ManifoldRenderer;

// Orchestrates the multi-panel interactive visualization
// Layout: [Volume 3D | Persistence Diagram] + [Latent Manifold] bottom panel
// All views are linked: selecting a feature in one highlights it in the others
class RenderEngine {
public:
    struct Config {
        int   width  = 1920;
        int   height = 1080;
        bool  vsync  = true;
        bool  msaa4x = true;
        float fov    = 45.0f;
    };

    explicit RenderEngine(const Config& cfg = {});
    ~RenderEngine();

    void init();
    void run();   // enters main render loop
    void shutdown();

    // Inject renderers
    void setVolumeRenderer  (std::shared_ptr<VolumeRenderer>   r);
    void setDiagramRenderer (std::shared_ptr<DiagramRenderer>  r);
    void setManifoldRenderer(std::shared_ptr<ManifoldRenderer> r);

    // Callback when user clicks a persistence point in the diagram
    // Receives (birth, death, dimension) → callers can highlight corresponding tissue
    using ClickCallback = std::function<void(float birth, float death, int dim)>;
    void onDiagramClick(ClickCallback cb);

    bool isRunning() const;

private:
    Config       m_cfg;
    GLFWwindow*  m_window = nullptr;

    std::shared_ptr<VolumeRenderer>   m_volume;
    std::shared_ptr<DiagramRenderer>  m_diagram;
    std::shared_ptr<ManifoldRenderer> m_manifold;

    ClickCallback m_click_cb;

    void processInput();
    void renderFrame();
};

} // namespace mmviz::render
