#include "render/RenderEngine.h"
#include "render/VolumeRenderer.h"
#include "render/DiagramRenderer.h"
#include "render/ManifoldRenderer.h"
#include "utils/Logger.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>

namespace mmviz::render {

RenderEngine::RenderEngine(const RenderConfig& cfg) : m_cfg(cfg) {}

RenderEngine::~RenderEngine() { shutdown(); }

void RenderEngine::init() {
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (m_cfg.msaa4x) glfwWindowHint(GLFW_SAMPLES, 4);

    m_window = glfwCreateWindow(m_cfg.width, m_cfg.height,
                                "MammoViz-TDA — Breast Cancer Topological Visualizer",
                                nullptr, nullptr);
    if (!m_window)
        throw std::runtime_error("glfwCreateWindow failed");

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(m_cfg.vsync ? 1 : 0);

    if (glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    if (m_cfg.msaa4x) glEnable(GL_MULTISAMPLE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    LOG_INFO("OpenGL {}, renderer: {}",
             (const char*)glGetString(GL_VERSION),
             (const char*)glGetString(GL_RENDERER));
}

void RenderEngine::run() {
    while (!glfwWindowShouldClose(m_window)) {
        processInput();
        renderFrame();
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void RenderEngine::renderFrame() {
    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Identity matrices for now — camera system implemented in Phase 2
    float view[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-3,1};
    float proj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    int W = m_cfg.width, H = m_cfg.height;
    int half_w = W / 2;

    // Left panel: volume
    if (m_volume) {
        glViewport(0, H/3, half_w, 2*H/3);
        m_volume->render(view, proj);
    }

    // Right panel: persistence diagram
    if (m_diagram) {
        glViewport(half_w, H/3, half_w, 2*H/3);
        m_diagram->render(half_w, H/3, half_w, 2*H/3);
    }

    // Bottom panel: latent manifold
    if (m_manifold) {
        glViewport(0, 0, W, H/3);
        m_manifold->render(0, 0, W, H/3, view, proj);
    }
}

void RenderEngine::processInput() {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, true);
}

void RenderEngine::shutdown() {
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

void RenderEngine::setVolumeRenderer  (std::shared_ptr<VolumeRenderer>   r) { m_volume   = r; }
void RenderEngine::setDiagramRenderer (std::shared_ptr<DiagramRenderer>  r) { m_diagram  = r; }
void RenderEngine::setManifoldRenderer(std::shared_ptr<ManifoldRenderer> r) { m_manifold = r; }

void RenderEngine::onDiagramClick(ClickCallback cb) { m_click_cb = std::move(cb); }

bool RenderEngine::isRunning() const {
    return m_window && !glfwWindowShouldClose(m_window);
}

} // namespace mmviz::render
