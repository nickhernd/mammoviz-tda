#include "render/RenderEngine.h"
#include "render/VolumeRenderer.h"
#include "render/DiagramRenderer.h"
#include "render/ManifoldRenderer.h"
#include "utils/Logger.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <cmath>
#include <cstring>

#ifdef HAVE_IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif

namespace mmviz::render {

// ── Helpers ───────────────────────────────────────────────────────────────────

static RenderEngine* g_engine = nullptr;  // for GLFW callbacks

static float deg2rad(float d) { return d * 3.14159265f / 180.f; }

// Build camera position from orbit params (azimuth, elevation, distance)
// around the cube center (0.5, 0.5, 0.5)
static void orbitCamera(float az_deg, float el_deg, float dist,
                        float& ex, float& ey, float& ez)
{
    float az = deg2rad(az_deg);
    float el = deg2rad(el_deg);
    ex = 0.5f + dist * std::cos(el) * std::sin(az);
    ey = 0.5f + dist * std::sin(el);
    ez = 0.5f + dist * std::cos(el) * std::cos(az);
}

// ── Construction ──────────────────────────────────────────────────────────────

RenderEngine::RenderEngine(const RenderConfig& cfg) : m_cfg(cfg) { g_engine = this; }

RenderEngine::~RenderEngine() { shutdown(); g_engine = nullptr; }

// ── GLFW callbacks ────────────────────────────────────────────────────────────

void RenderEngine::mouseButtonCb(GLFWwindow* w, int btn, int action, int /*mods*/) {
    if (!g_engine) return;
#ifdef HAVE_IMGUI
    if (ImGui::GetIO().WantCaptureMouse) return;
#endif
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        g_engine->m_dragging = (action == GLFW_PRESS);
        if (g_engine->m_dragging)
            glfwGetCursorPos(w, &g_engine->m_last_mx, &g_engine->m_last_my);
    }
}

void RenderEngine::cursorPosCb(GLFWwindow* /*w*/, double mx, double my) {
    if (!g_engine || !g_engine->m_dragging) return;
    double dx = mx - g_engine->m_last_mx;
    double dy = my - g_engine->m_last_my;
    g_engine->m_last_mx = mx;
    g_engine->m_last_my = my;
    g_engine->m_cam_azimuth   += (float)dx * 0.4f;
    g_engine->m_cam_elevation -= (float)dy * 0.3f;
    g_engine->m_cam_elevation  = std::max(-89.f, std::min(89.f, g_engine->m_cam_elevation));
}

void RenderEngine::scrollCb(GLFWwindow* /*w*/, double /*dx*/, double dy) {
    if (!g_engine) return;
#ifdef HAVE_IMGUI
    if (ImGui::GetIO().WantCaptureMouse) return;
#endif
    g_engine->m_cam_distance -= (float)dy * 0.15f;
    g_engine->m_cam_distance  = std::max(0.5f, std::min(8.f, g_engine->m_cam_distance));
}

// ── Init / Shutdown ───────────────────────────────────────────────────────────

void RenderEngine::init() {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);

    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(m_cfg.width, m_cfg.height,
                                "MammoViz-TDA  |  Breast Cancer AI Visualizer",
                                nullptr, nullptr);
    if (!m_window)
        throw std::runtime_error("glfwCreateWindow failed");

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(m_cfg.vsync ? 1 : 0);

    glfwSetMouseButtonCallback(m_window, mouseButtonCb);
    glfwSetCursorPosCallback  (m_window, cursorPosCb);
    glfwSetScrollCallback     (m_window, scrollCb);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
        throw std::runtime_error(std::string("glewInit failed: ") +
                                 reinterpret_cast<const char*>(glewGetErrorString(err)));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    LOG_INFO("OpenGL {}, renderer: {}",
             reinterpret_cast<const char*>(glGetString(GL_VERSION)),
             reinterpret_cast<const char*>(glGetString(GL_RENDERER)));

    initImGui();
}

void RenderEngine::initImGui() {
#ifdef HAVE_IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Style: dark clinical look
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding    = 6.f;
    st.FrameRounding     = 4.f;
    st.ItemSpacing       = {8, 5};
    st.WindowPadding     = {10, 10};
    st.Colors[ImGuiCol_WindowBg]      = {0.10f, 0.12f, 0.16f, 0.92f};
    st.Colors[ImGuiCol_TitleBg]       = {0.08f, 0.10f, 0.18f, 1.f};
    st.Colors[ImGuiCol_TitleBgActive] = {0.12f, 0.18f, 0.32f, 1.f};
    st.Colors[ImGuiCol_FrameBg]       = {0.16f, 0.18f, 0.24f, 1.f};
    st.Colors[ImGuiCol_Button]        = {0.20f, 0.30f, 0.50f, 1.f};
    st.Colors[ImGuiCol_ButtonHovered] = {0.28f, 0.42f, 0.68f, 1.f};
    st.Colors[ImGuiCol_Header]        = {0.18f, 0.26f, 0.42f, 1.f};

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 460 core");
    m_imgui_initialized = true;
    LOG_INFO("ImGui initialized");
#endif
}

void RenderEngine::shutdownImGui() {
#ifdef HAVE_IMGUI
    if (!m_imgui_initialized) return;
    m_imgui_initialized = false;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif
}

void RenderEngine::shutdown() {
    shutdownImGui();
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

// ── State setters ─────────────────────────────────────────────────────────────

void RenderEngine::setAppState(const AppState& s) { m_state = s; }

void RenderEngine::setVolumeRenderer  (std::shared_ptr<VolumeRenderer>   r) { m_volume   = r; }
void RenderEngine::setDiagramRenderer (std::shared_ptr<DiagramRenderer>  r) { m_diagram  = r; }
void RenderEngine::setManifoldRenderer(std::shared_ptr<ManifoldRenderer> r) { m_manifold = r; }
void RenderEngine::onDiagramClick(ClickCallback cb) { m_click_cb = std::move(cb); }
bool RenderEngine::isRunning() const { return m_window && !glfwWindowShouldClose(m_window); }

// ── Main loop ─────────────────────────────────────────────────────────────────

void RenderEngine::run() {
    while (!glfwWindowShouldClose(m_window)) {
        processInput();
        renderFrame();
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void RenderEngine::processInput() {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, true);
    if (glfwGetKey(m_window, GLFW_KEY_G) == GLFW_PRESS)
        m_show_gradcam = !m_show_gradcam;
    if (glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS) {
        m_cam_azimuth = 0.f; m_cam_elevation = 20.f; m_cam_distance = 2.5f;
    }
}

// ── Frame rendering ───────────────────────────────────────────────────────────

void RenderEngine::renderFrame() {
    glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int W = m_cfg.width, H = m_cfg.height;

    // Sidebar width (ImGui panel on the right)
    constexpr int SIDEBAR = 260;
    int render_w = W - SIDEBAR;
    int half_w   = render_w / 2;

    // Build orbit camera for volume
    float ex, ey, ez;
    orbitCamera(m_cam_azimuth, m_cam_elevation, m_cam_distance, ex, ey, ez);
    if (m_volume) {
        m_volume->setCameraPos(ex, ey, ez);
        m_volume->setGradCAM(m_show_gradcam);
        m_layers.lesion_visible = m_show_gradcam;
        m_volume->setLayerParams(m_layers);
    }

    // Left panel: volume (top 2/3 of render area)
    if (m_volume) {
        glViewport(0, H / 3, half_w, 2 * H / 3);
        float dummy[16] = {};
        m_volume->render(dummy, dummy);
    }

    // Right panel: persistence diagram
    if (m_diagram) {
        glViewport(half_w, H / 3, half_w, 2 * H / 3);
        m_diagram->render(half_w, H / 3, half_w, 2 * H / 3);
    }

    // Bottom strip: manifold (or info strip if no data)
    if (m_manifold) {
        glViewport(0, 0, render_w, H / 3);
        float dummy[16] = {};
        m_manifold->render(0, 0, render_w, H / 3, dummy, dummy);
    }

    // Reset viewport before ImGui
    glViewport(0, 0, W, H);

#ifdef HAVE_IMGUI
    renderImGui();
#endif
}

// ── ImGui overlay ─────────────────────────────────────────────────────────────

void RenderEngine::renderImGui() {
#ifdef HAVE_IMGUI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int W = m_cfg.width, H = m_cfg.height;
    constexpr int SIDEBAR = 260;

    // ── Sidebar panel ─────────────────────────────────────────────────────────
    ImGui::SetNextWindowPos ({(float)(W - SIDEBAR), 0}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({(float)SIDEBAR, (float)H}, ImGuiCond_Always);
    ImGui::Begin("##sidebar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar);

    // Title
    ImGui::PushStyleColor(ImGuiCol_Text, {0.5f, 0.8f, 1.0f, 1.f});
    ImGui::SetWindowFontScale(1.2f);
    ImGui::Text("MammoViz-TDA");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::TextDisabled("Breast Cancer AI");
    ImGui::Separator();

    // ── Prediction block ──────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::TextDisabled("PREDICTION");
    ImGui::Spacing();

    if (m_state.pred_class < 0) {
        ImGui::TextDisabled("No model loaded");
    } else {
        const bool is_malignant = (m_state.pred_class == 1);
        const char* label = is_malignant ? "MALIGNANT" : "BENIGN";

        // Big colored label
        ImVec4 col = is_malignant
            ? ImVec4(1.0f, 0.25f, 0.20f, 1.f)   // red for malignant
            : ImVec4(0.25f, 0.90f, 0.40f, 1.f);  // green for benign
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::SetWindowFontScale(1.6f);
        float tw = ImGui::CalcTextSize(label).x;
        ImGui::SetCursorPosX((SIDEBAR - tw) * 0.5f);
        ImGui::Text("%s", label);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();

        // Confidence bar
        ImGui::Spacing();
        float conf = m_state.confidence;
        char bar_label[32];
        snprintf(bar_label, sizeof(bar_label), "Confidence: %.1f%%", conf * 100.f);
        ImVec4 bar_col = is_malignant ? ImVec4(0.9f,0.2f,0.15f,1.f) : ImVec4(0.2f,0.75f,0.3f,1.f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_col);
        ImGui::ProgressBar(conf, {-1, 0}, bar_label);
        ImGui::PopStyleColor();

        // Logits
        ImGui::Spacing();
        ImGui::Text("Benign:    %.3f", m_state.logit_benign);
        ImGui::Text("Malignant: %.3f", m_state.logit_malignant);
    }

    ImGui::Separator();

    // ── TDA stats ─────────────────────────────────────────────────────────────
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Topological Features", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("H0 (components): %d", m_state.tda_h0);
        ImGui::Text("H1 (loops):      %d", m_state.tda_h1);
        ImGui::Text("H2 (voids):      %d", m_state.tda_h2);
        ImGui::Text("Total pairs:     %d", m_state.tda_total_points);
        ImGui::Text("Max persistence: %.3f", m_state.tda_max_persistence);
        ImGui::Spacing();
        ImGui::TextDisabled("Drag diagram to explore");
    }

    ImGui::Separator();
    ImGui::Spacing();

    // ── Volume info ───────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Volume Info")) {
        if (!m_state.case_id.empty())
            ImGui::Text("Case: %s", m_state.case_id.c_str());
        ImGui::Text("Size: %dx%dx%d", m_state.vol_x, m_state.vol_y, m_state.vol_z);
        ImGui::Text("Spacing XY: %.3f mm", m_state.spacing_xy);
        ImGui::Text("Spacing Z:  %.3f mm", m_state.spacing_z);
        if (!m_state.pattern.empty()) {
            ImGui::TextDisabled("Ground truth: %s", m_state.pattern.c_str());
        }
    }

    ImGui::Separator();
    ImGui::Spacing();

    // ── Anatomical layers ─────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Anatomical Layers", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f,0.85f,1.0f,1.f));
        ImGui::Text("Layer          Vis  Opacity");
        ImGui::PopStyleColor();
        ImGui::Separator();

        // Helper lambda: one row per layer
        auto layerRow = [&](const char* name, ImVec4 dot_col,
                            bool& vis, float& opacity, float r, float g, float b) {
            ImGui::PushID(name);
            // Colored dot
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddCircleFilled(
                {p.x + 6, p.y + 8}, 5, IM_COL32(
                    (int)(r*255),(int)(g*255),(int)(b*255),255));
            ImGui::Dummy({14, 0}); ImGui::SameLine();
            ImGui::Text("%-13s", name); ImGui::SameLine();
            ImGui::Checkbox("##v", &vis); ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            ImGui::SliderFloat("##o", &opacity, 0.f, 1.f, "%.2f");
            ImGui::PopID();
        };

        auto& ly = m_layers;
        layerRow("Fat",        {}, ly.fat.visible,   ly.fat.opacity,   0.95f,0.82f,0.35f);
        layerRow("Glandular",  {}, ly.gland.visible, ly.gland.opacity, 0.85f,0.45f,0.55f);
        layerRow("Dense",      {}, ly.dense.visible, ly.dense.opacity, 1.00f,0.78f,0.55f);
        layerRow("Calcif.",    {}, ly.calc.visible,  ly.calc.opacity,  1.00f,0.97f,0.65f);

        ImGui::Spacing();
        ImGui::Separator();
        // GradCAM / lesion overlay row
        ImGui::PushID("lesion");
        ImVec2 p2 = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(
            {p2.x+6,p2.y+8}, 5, IM_COL32(255,40,20,255));
        ImGui::Dummy({14,0}); ImGui::SameLine();
        ImGui::Text("%-13s", "Lesion/CAM"); ImGui::SameLine();
        ImGui::Checkbox("##v", &m_show_gradcam); ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("##o", &ly.lesion_opacity, 0.f, 1.f, "%.2f");
        ImGui::PopID();

        ImGui::Spacing();
        ImGui::Text("CAM threshold:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##sal", &ly.lesion_sal_thresh, 0.f, 0.8f, "%.2f");
    }

    ImGui::Separator();
    ImGui::Spacing();

    // ── Clip planes ───────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Clip Planes (Peel)")) {
        ImGui::TextDisabled("Slide left to peel away layers");
        ImGui::Spacing();
        ImGui::Text("X (Left→Right)");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##cx", &m_layers.clip_x, 0.1f, 1.f, "%.2f");
        ImGui::Text("Y (Bot→Top)");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##cy", &m_layers.clip_y, 0.1f, 1.f, "%.2f");
        ImGui::Text("Z (Front→Back)");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##cz", &m_layers.clip_z, 0.1f, 1.f, "%.2f");
        if (ImGui::Button("Reset planes", {-1,0})) {
            m_layers.clip_x = m_layers.clip_y = m_layers.clip_z = 1.f;
        }
    }

    ImGui::Separator();
    ImGui::Spacing();

    // ── Camera ────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Camera")) {
        ImGui::Text("Az: %.1f  El: %.1f", m_cam_azimuth, m_cam_elevation);
        ImGui::Text("Dist: %.2f", m_cam_distance);
        if (ImGui::Button("Reset [R]", {-1, 0}))
            m_cam_azimuth = 0.f, m_cam_elevation = 20.f, m_cam_distance = 2.5f;
    }

    ImGui::Separator();
    ImGui::Spacing();

    // ── Shortcuts footer ──────────────────────────────────────────────────────
    ImGui::TextDisabled("ESC  Exit");
    ImGui::TextDisabled("G    Toggle lesion");
    ImGui::TextDisabled("R    Reset camera");

    ImGui::End();

    // ── Panel labels (small overlay text) ────────────────────────────────────
    ImGuiWindowFlags label_flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove   |
        ImGuiWindowFlags_NoSavedSettings;

    int render_w = W - SIDEBAR;
    int half_w   = render_w / 2;

    ImGui::SetNextWindowPos({4, 4}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({(float)half_w - 8, 22}, ImGuiCond_Always);
    ImGui::Begin("##vol_label", nullptr, label_flags);
    ImGui::TextColored({0.6f,0.8f,1.f,0.8f}, "Volume (Ray Casting)");
    ImGui::End();

    ImGui::SetNextWindowPos({(float)half_w + 4, 4}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({(float)half_w - 8, 22}, ImGuiCond_Always);
    ImGui::Begin("##diag_label", nullptr, label_flags);
    ImGui::TextColored({0.5f,1.f,0.6f,0.8f}, "Persistence Diagram (TDA)");
    ImGui::End();

    ImGui::SetNextWindowPos({4, (float)(2 * H / 3) + 4}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({(float)render_w - 8, 22}, ImGuiCond_Always);
    ImGui::Begin("##mfld_label", nullptr, label_flags);
    ImGui::TextColored({1.f,0.8f,0.5f,0.8f}, "Latent Manifold (UMAP/PCA)");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif
}

} // namespace mmviz::render
