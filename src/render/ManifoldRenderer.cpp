#include "render/ManifoldRenderer.h"
#include "utils/Logger.h"
#include <GL/glew.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

GLuint linkProgram(const std::string& vp, const std::string& fp) {
    GLuint vs   = compileShader(GL_VERTEX_SHADER,   readFile(vp));
    GLuint fs   = compileShader(GL_FRAGMENT_SHADER, readFile(fp));
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

// Column-major perspective
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
    float zx = ex-tx, zy = ey-ty, zz = ez-tz;
    float zl = std::sqrt(zx*zx + zy*zy + zz*zz);
    zx /= zl; zy /= zl; zz /= zl;

    float xx = zz, xz = -zx;
    float xl = std::sqrt(xx*xx + xz*xz);
    if (xl > 1e-6f) { xx /= xl; xz /= xl; }
    float xy = 0.0f;

    float yx = zy*xz - zz*xy;
    float yy = zz*xx - zx*xz;
    float yz = zx*xy - zy*xx;

    memset(m, 0, 64);
    m[0]=xx;  m[4]=xy;  m[8] =xz;  m[12]=-(xx*ex + xy*ey + xz*ez);
    m[1]=yx;  m[5]=yy;  m[9] =yz;  m[13]=-(yx*ex + yy*ey + yz*ez);
    m[2]=zx;  m[6]=zy;  m[10]=zz;  m[14]=-(zx*ex + zy*ey + zz*ez);
    m[3]=0;   m[7]=0;   m[11]=0;   m[15]=1;
}

// c = a * b, column major
void mat4_mul(float* c, const float* a, const float* b) {
    memset(c, 0, 64);
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                c[col*4+row] += a[k*4+row] * b[col*4+k];
}

// VBO layout: {x, y, z, confidence, label (int), selected (float)}
struct ManifoldVertex {
    float   x, y, z;
    float   confidence;
    int32_t label;
    float   selected;
};

} // anonymous namespace

namespace mmviz::render {

ManifoldRenderer::ManifoldRenderer() {}

ManifoldRenderer::~ManifoldRenderer() {
    if (m_vao)    glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)    glDeleteBuffers(1, &m_vbo);
    if (m_shader) glDeleteProgram(m_shader);
}

void ManifoldRenderer::setPoints(const std::vector<xai::ProjectedPoint>& pts) {
    m_points = pts;
    m_pts_dirty = true;
    LOG_INFO("Manifold: {} projected samples", pts.size());
}

void ManifoldRenderer::setSelectedSample(const std::string& id) {
    m_selected_id = id;
    m_pts_dirty = true;
}

void ManifoldRenderer::showTrajectories(bool enabled) {
    m_trajectories = enabled;
}

// ── GL resource initialization ────────────────────────────────────────────────

void ManifoldRenderer::initShaders() {
    try {
        m_shader = linkProgram("shaders/glsl/manifold.vert",
                               "shaders/glsl/manifold.frag");
    } catch (const std::exception& e) {
        LOG_ERROR("ManifoldRenderer shader: {}", e.what());
        return;
    }

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    constexpr GLsizei stride = sizeof(ManifoldVertex);
    // location 0: vec3 position
    glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(ManifoldVertex, x));
    glEnableVertexAttribArray(0);
    // location 1: float confidence
    glVertexAttribPointer (1, 1, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(ManifoldVertex, confidence));
    glEnableVertexAttribArray(1);
    // location 2: int label (integer attribute)
    glVertexAttribIPointer(2, 1, GL_INT,             stride, (void*)offsetof(ManifoldVertex, label));
    glEnableVertexAttribArray(2);
    // location 3: float selected
    glVertexAttribPointer (3, 1, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(ManifoldVertex, selected));
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);

    m_shaders_ok = true;
    LOG_INFO("ManifoldRenderer: GL resources initialized");
}

void ManifoldRenderer::uploadPoints() {
    m_num_points = 0;
    if (m_points.empty()) { m_pts_dirty = false; return; }

    std::vector<ManifoldVertex> verts;
    verts.reserve(m_points.size());
    for (const auto& p : m_points) {
        float sel = (p.sample_id == m_selected_id && !m_selected_id.empty()) ? 1.0f : 0.0f;
        verts.push_back({ p.coords[0], p.coords[1], p.coords[2],
                          p.confidence, p.label, sel });
    }
    m_num_points = (int)verts.size();

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(verts.size() * sizeof(ManifoldVertex)),
                 verts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    m_pts_dirty = false;
}

// ── Frame render ──────────────────────────────────────────────────────────────

void ManifoldRenderer::render(int /*px*/, int /*py*/, int /*pw*/, int /*ph*/,
                               const float* /*ext_view*/, const float* /*ext_proj*/)
{
    if (!m_shaders_ok) initShaders();
    if (!m_shaders_ok) return;
    if (m_pts_dirty)   uploadPoints();
    if (m_num_points == 0) return;

    // Fixed camera orbiting the origin — points are in [-1,1]^3
    float view[16], proj[16], mvp[16];
    buildPerspective(proj, 0.7854f /*45°*/, 16.0f / 9.0f, 0.1f, 20.0f);
    buildLookAt(view, 0.0f, 0.0f, 4.0f,
                       0.0f, 0.0f, 0.0f);
    mat4_mul(mvp, proj, view);  // model = identity

    glUseProgram(m_shader);
    glUniformMatrix4fv(glGetUniformLocation(m_shader, "u_mvp"), 1, GL_FALSE, mvp);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_num_points);
    glBindVertexArray(0);
    glUseProgram(0);
}

std::string ManifoldRenderer::hitTest(float, float, float, float, float, float) const {
    return {};
}

} // namespace mmviz::render
