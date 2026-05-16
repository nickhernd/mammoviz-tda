#include "render/VolumeRenderer.h"
#include "utils/Logger.h"
#include <GL/glew.h>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open shader: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLuint compileShader(GLenum type, const std::string& src) {
    const char* code = src.c_str();
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &code, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        glDeleteShader(s);
        throw std::runtime_error(std::string("Shader compile: ") + log);
    }
    return s;
}

GLuint linkProgram(const std::string& vert_path, const std::string& frag_path) {
    GLuint vs = compileShader(GL_VERTEX_SHADER,   readFile(vert_path));
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, readFile(frag_path));
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Program link: ") + log);
    }
    return prog;
}

// Column-major perspective matrix (standard OpenGL)
void buildPerspective(float* m, float fov_rad, float aspect, float n, float f) {
    memset(m, 0, 64);
    float t = 1.0f / std::tan(fov_rad * 0.5f);
    m[0]  = t / aspect;
    m[5]  = t;
    m[10] = (f + n) / (n - f);
    m[11] = -1.0f;
    m[14] = 2.0f * f * n / (n - f);
}

// Column-major look-at (world_up = (0,1,0) assumed)
void buildLookAt(float* m,
                 float ex, float ey, float ez,
                 float tx, float ty, float tz)
{
    // z_cam = normalize(eye - target)  — points away from scene
    float zx = ex-tx, zy = ey-ty, zz = ez-tz;
    float zl = std::sqrt(zx*zx + zy*zy + zz*zz);
    zx /= zl; zy /= zl; zz /= zl;

    // x_cam = normalize(cross(world_up=(0,1,0), z_cam))
    float xx = zz, xz = -zx;  // cross((0,1,0),(zx,zy,zz)).xz only (y=0)
    float xl = std::sqrt(xx*xx + xz*xz);
    xx /= xl; xz /= xl;
    float xy = 0.0f;

    // y_cam = cross(z_cam, x_cam)
    float yx = zy*xz - zz*xy;
    float yy = zz*xx - zx*xz;
    float yz = zx*xy - zy*xx;

    memset(m, 0, 64);
    // Column major: m[col*4+row]
    m[0]=xx;  m[4]=xy;  m[8] =xz;  m[12]=-(xx*ex + xy*ey + xz*ez);
    m[1]=yx;  m[5]=yy;  m[9] =yz;  m[13]=-(yx*ex + yy*ey + yz*ez);
    m[2]=zx;  m[6]=zy;  m[10]=zz;  m[14]=-(zx*ex + zy*ey + zz*ez);
    m[3]=0;   m[7]=0;   m[11]=0;   m[15]=1;
}

// Unit cube [0,1]^3
static const float kCubeVerts[] = {
    0,0,0,  1,0,0,  1,1,0,  0,1,0,   // back  (z=0)
    0,0,1,  1,0,1,  1,1,1,  0,1,1,   // front (z=1)
};
static const unsigned int kCubeIdx[] = {
    0,2,1, 0,3,2,   // back
    4,5,6, 4,6,7,   // front
    0,4,7, 0,7,3,   // left
    1,2,6, 1,6,5,   // right
    0,1,5, 0,5,4,   // bottom
    3,7,6, 3,6,2,   // top
};

} // anonymous namespace

namespace mmviz::render {

// ── Transfer function presets ─────────────────────────────────────────────────

VolumeRenderer::TransferFunction VolumeRenderer::TransferFunction::breastDefault() {
    TransferFunction tf;
    tf.lut.resize(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float a = (t < 0.1f) ? 0.0f : (t < 0.5f) ? (t - 0.1f) * 0.5f : t * 0.8f;
        tf.lut[i] = { t, t * 0.9f, t * 0.7f, a };
    }
    return tf;
}

VolumeRenderer::TransferFunction VolumeRenderer::TransferFunction::calcificationHighlight() {
    TransferFunction tf;
    tf.lut.resize(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float a = (t < 0.8f) ? 0.01f : (t - 0.8f) * 4.0f;
        float r = (t > 0.8f) ? 1.0f : t * 0.3f;
        tf.lut[i] = { r, t * 0.3f, t * 0.3f, a };
    }
    return tf;
}

// ── Construction / destruction ────────────────────────────────────────────────

VolumeRenderer::VolumeRenderer() {
    m_tf = TransferFunction::breastDefault();
}

VolumeRenderer::~VolumeRenderer() {
    if (m_vol_tex) glDeleteTextures(1, &m_vol_tex);
    if (m_sal_tex) glDeleteTextures(1, &m_sal_tex);
    if (m_tf_tex)  glDeleteTextures(1, &m_tf_tex);
    if (m_vbo)     glDeleteBuffers(1, &m_vbo);
    if (m_ebo)     glDeleteBuffers(1, &m_ebo);
    if (m_vao)     glDeleteVertexArrays(1, &m_vao);
    if (m_shader)  glDeleteProgram(m_shader);
}

// ── Lazy GL initialization (must be called with active context) ───────────────

void VolumeRenderer::lazyInit() {
    try {
        m_shader = linkProgram("shaders/glsl/volume_raycast.vert",
                               "shaders/glsl/volume_raycast.frag");
    } catch (const std::exception& e) {
        LOG_ERROR("VolumeRenderer shader: {}", e.what());
        return;
    }

    // Unit cube geometry
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kCubeVerts), kCubeVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kCubeIdx), kCubeIdx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    m_num_idx = 36;

    // 1D transfer function texture
    glGenTextures(1, &m_tf_tex);
    glBindTexture(GL_TEXTURE_1D, m_tf_tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, 256, 0,
                 GL_RGBA, GL_FLOAT, m_tf.lut.data());
    glBindTexture(GL_TEXTURE_1D, 0);

    m_ready = true;
    LOG_INFO("VolumeRenderer: GL resources initialized");
}

// ── Data upload ───────────────────────────────────────────────────────────────

void VolumeRenderer::uploadVolume(const io::VolumeData& vol) {
    if (m_vol_tex) glDeleteTextures(1, &m_vol_tex);
    glGenTextures(1, &m_vol_tex);
    glBindTexture(GL_TEXTURE_3D, m_vol_tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                 vol.shape[0], vol.shape[1], vol.shape[2],
                 0, GL_RED, GL_FLOAT, vol.data.data());
    m_vol_dims = {vol.shape[0], vol.shape[1], vol.shape[2]};
    LOG_INFO("Volume uploaded to GPU: {}×{}×{}", vol.shape[0], vol.shape[1], vol.shape[2]);
}

void VolumeRenderer::uploadSaliency(const xai::GradCAM::SaliencyMap& map) {
    if (map.values.empty()) return;
    if (m_sal_tex) glDeleteTextures(1, &m_sal_tex);
    glGenTextures(1, &m_sal_tex);
    glBindTexture(GL_TEXTURE_3D, m_sal_tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                 map.shape[0], map.shape[1], map.shape[2],
                 0, GL_RED, GL_FLOAT, map.values.data());
}

void VolumeRenderer::setRenderMode(RenderMode mode) { m_mode = mode; }
void VolumeRenderer::setCameraPos(float x, float y, float z) { m_cam_x=x; m_cam_y=y; m_cam_z=z; }
void VolumeRenderer::setGradCAM(bool e) { m_gradcam_on = e; }
void VolumeRenderer::setLayerParams(const LayerParams& lp) { m_layers = lp; }

void VolumeRenderer::setTransferFunction(const TransferFunction& tf) {
    m_tf = tf;
    if (m_tf_tex) {
        glBindTexture(GL_TEXTURE_1D, m_tf_tex);
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGBA, GL_FLOAT, tf.lut.data());
        glBindTexture(GL_TEXTURE_1D, 0);
    }
}

// ── Frame render ─────────────────────────────────────────────────────────────

void VolumeRenderer::render(const float* /*ext_view*/, const float* /*ext_proj*/) {
    if (!m_vol_tex) return;
    if (!m_ready)   lazyInit();
    if (!m_shader || !m_vao) return;

    // Fixed camera: eye at (0.5, 0.5, 4.0) looking at cube center (0.5, 0.5, 0.5)
    float cam_view[16], cam_proj[16];
    buildPerspective(cam_proj, 0.9599f /*55°*/, 4.0f / 3.0f, 0.01f, 20.0f);
    buildLookAt(cam_view, m_cam_x, m_cam_y, m_cam_z,
                          0.5f,    0.5f,    0.5f);

    float identity[16];
    memset(identity, 0, sizeof(identity));
    identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;

    glUseProgram(m_shader);

    glUniformMatrix4fv(glGetUniformLocation(m_shader, "u_model"), 1, GL_FALSE, identity);
    glUniformMatrix4fv(glGetUniformLocation(m_shader, "u_view"),  1, GL_FALSE, cam_view);
    glUniformMatrix4fv(glGetUniformLocation(m_shader, "u_proj"),  1, GL_FALSE, cam_proj);
    glUniform3f(glGetUniformLocation(m_shader, "u_camera_pos"), m_cam_x, m_cam_y, m_cam_z);
    glUniform1f(glGetUniformLocation(m_shader, "u_step_size"), 0.003f);
    glUniform3f(glGetUniformLocation(m_shader, "u_vol_size"),
                (float)m_vol_dims[0], (float)m_vol_dims[1], (float)m_vol_dims[2]);
    glUniform1i(glGetUniformLocation(m_shader, "u_show_gradcam"),
                (m_sal_tex != 0 && m_gradcam_on) ? 1 : 0);

    // ── Layer uniforms ────────────────────────────────────────────────────────
    auto ul = [&](const char* n, GLint v){ glUniform1i(glGetUniformLocation(m_shader,n),v); };
    auto uf = [&](const char* n, GLfloat v){ glUniform1f(glGetUniformLocation(m_shader,n),v); };
    auto u3 = [&](const char* n, float r, float g, float b){
        glUniform3f(glGetUniformLocation(m_shader,n), r, g, b); };

    const auto& ly = m_layers;
    u3("u_fat_color",   ly.fat.r,   ly.fat.g,   ly.fat.b);
    uf("u_fat_opacity", ly.fat.opacity);
    ul("u_fat_visible", ly.fat.visible ? 1 : 0);

    u3("u_gland_color",   ly.gland.r,   ly.gland.g,   ly.gland.b);
    uf("u_gland_opacity", ly.gland.opacity);
    ul("u_gland_visible", ly.gland.visible ? 1 : 0);

    u3("u_dense_color",   ly.dense.r,   ly.dense.g,   ly.dense.b);
    uf("u_dense_opacity", ly.dense.opacity);
    ul("u_dense_visible", ly.dense.visible ? 1 : 0);

    u3("u_calc_color",   ly.calc.r,   ly.calc.g,   ly.calc.b);
    uf("u_calc_opacity", ly.calc.opacity);
    ul("u_calc_visible", ly.calc.visible ? 1 : 0);

    uf("u_lesion_sal_thresh", ly.lesion_sal_thresh);
    uf("u_lesion_opacity",    ly.lesion_opacity);
    ul("u_lesion_visible",    ly.lesion_visible ? 1 : 0);

    uf("u_clip_x", ly.clip_x);
    uf("u_clip_y", ly.clip_y);
    uf("u_clip_z", ly.clip_z);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_vol_tex);
    glUniform1i(glGetUniformLocation(m_shader, "u_volume"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, m_sal_tex ? m_sal_tex : m_vol_tex);
    glUniform1i(glGetUniformLocation(m_shader, "u_saliency"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_1D, m_tf_tex);
    glUniform1i(glGetUniformLocation(m_shader, "u_transfer"), 2);

    // Volume is transparent: cull back faces, disable depth write,
    // use pre-multiplied alpha blend (ONE / ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, m_num_idx, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glUseProgram(0);

    // Restore global state
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

std::vector<std::array<int,3>> VolumeRenderer::voxelsForPersistenceRegion(
    float, float, int) const
{
    // Mapping implemented in Phase 5 (requires VietorisRips simplex→voxel table)
    return {};
}

} // namespace mmviz::render
