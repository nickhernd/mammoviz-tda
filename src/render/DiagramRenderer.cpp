#include "render/DiagramRenderer.h"
#include "utils/Logger.h"
#include <GL/glew.h>
#include <algorithm>
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

GLuint linkProgramFiles(const std::string& vp, const std::string& fp) {
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

GLuint linkProgramSrc(const char* vert_src, const char* frag_src) {
    GLuint vs   = compileShader(GL_VERTEX_SHADER,   vert_src);
    GLuint fs   = compileShader(GL_FRAGMENT_SHADER, frag_src);
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
        throw std::runtime_error(std::string("Program link (inline): ") + log);
    }
    return prog;
}

// Minimal 2D line shader for the persistence diagonal y = x
static const char* kLineVert = R"(
#version 460 core
layout(location=0) in vec2 a_pos;
uniform mat4 u_ortho;
void main() { gl_Position = u_ortho * vec4(a_pos, 0.0, 1.0); }
)";
static const char* kLineFrag = R"(
#version 460 core
out vec4 frag_color;
void main() { frag_color = vec4(0.55, 0.55, 0.55, 0.7); }
)";

// Column-major 2D orthographic projection: [left,right] × [bottom,top]
void buildOrtho2D(float* m, float l, float r, float b, float t) {
    memset(m, 0, 64);
    m[0]  = 2.0f / (r - l);
    m[5]  = 2.0f / (t - b);
    m[10] = -1.0f;
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[15] = 1.0f;
}

// VBO layout for diagram scatter: {birth, death, dimension, persistence}
struct DiagVertex {
    float   birth;
    float   death;
    int32_t dimension;
    float   persistence;
};

} // anonymous namespace

namespace mmviz::render {

DiagramRenderer::DiagramRenderer()  {}

DiagramRenderer::~DiagramRenderer() {
    if (m_vao)         glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)         glDeleteBuffers(1, &m_vbo);
    if (m_shader)      glDeleteProgram(m_shader);
    if (m_line_vao)    glDeleteVertexArrays(1, &m_line_vao);
    if (m_line_vbo)    glDeleteBuffers(1, &m_line_vbo);
    if (m_line_shader) glDeleteProgram(m_line_shader);
}

void DiagramRenderer::setDiagram(const tda::PersistenceDiagram& diagram) {
    m_current_diagram = diagram;
    m_scale_max = 1.0f;
    for (auto& p : diagram.points)
        m_scale_max = std::max(m_scale_max, p.death);
    m_pts_dirty = true;
    LOG_INFO("Diagram set: {} persistence points (max={:.3f})",
             diagram.points.size(), m_scale_max);
}

void DiagramRenderer::setTemporalDiagrams(const std::vector<tda::PersistenceDiagram>& diags) {
    m_temporal_diags = diags;
}

void DiagramRenderer::setCurrentPhase(int phase_index) {
    m_phase = phase_index;
    if (phase_index < (int)m_temporal_diags.size()) {
        setDiagram(m_temporal_diags[phase_index]);
    }
}

void DiagramRenderer::setMinPersistence(float min_p) {
    m_min_persistence = min_p;
}

// ── GL resource initialization ────────────────────────────────────────────────

void DiagramRenderer::initShaders() {
    try {
        m_shader = linkProgramFiles("shaders/glsl/diagram.vert",
                                    "shaders/glsl/diagram.frag");
    } catch (const std::exception& e) {
        LOG_ERROR("DiagramRenderer scatter shader: {}", e.what());
        return;
    }
    try {
        m_line_shader = linkProgramSrc(kLineVert, kLineFrag);
    } catch (const std::exception& e) {
        LOG_ERROR("DiagramRenderer line shader: {}", e.what());
        return;
    }

    // Scatter VAO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    // location 0: vec2 (birth, death)
    glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, sizeof(DiagVertex), (void*)0);
    glEnableVertexAttribArray(0);
    // location 1: int dimension (must use IPointer)
    glVertexAttribIPointer(1, 1, GL_INT, sizeof(DiagVertex), (void*)offsetof(DiagVertex, dimension));
    glEnableVertexAttribArray(1);
    // location 2: float persistence
    glVertexAttribPointer (2, 1, GL_FLOAT, GL_FALSE, sizeof(DiagVertex), (void*)offsetof(DiagVertex, persistence));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    // Line VAO (2 vertices: (0,0) and (max,max))
    glGenVertexArrays(1, &m_line_vao);
    glGenBuffers(1, &m_line_vbo);
    glBindVertexArray(m_line_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_line_vbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    m_shaders_ok = true;
    LOG_INFO("DiagramRenderer: GL resources initialized");
}

void DiagramRenderer::uploadPoints() {
    const auto& pts = m_current_diagram.points;
    m_num_points = 0;
    if (pts.empty()) { m_pts_dirty = false; return; }

    std::vector<DiagVertex> verts;
    verts.reserve(pts.size());
    for (const auto& p : pts) {
        if (p.persistence() < m_min_persistence) continue;
        verts.push_back({ p.birth, p.death, p.dimension, p.persistence() });
    }
    m_num_points = (int)verts.size();

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(verts.size() * sizeof(DiagVertex)),
                 verts.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(0);

    m_pts_dirty = false;
}

// ── Frame render ──────────────────────────────────────────────────────────────

void DiagramRenderer::render(int /*px*/, int /*py*/, int /*pw*/, int /*ph*/) {
    if (!m_shaders_ok) initShaders();
    if (!m_shaders_ok) return;
    if (m_pts_dirty)   uploadPoints();

    // Ortho: map [−pad, max+pad] to NDC, same scale on both axes
    float pad   = m_scale_max * 0.08f;
    float lo    = -pad;
    float hi    = m_scale_max + pad;

    float ortho[16];
    buildOrtho2D(ortho, lo, hi, lo, hi);

    // ── Draw diagonal y = x (reference: persistence = 0 at this line) ─────────
    {
        float line_pts[4] = { lo, lo, hi, hi };
        glBindBuffer(GL_ARRAY_BUFFER, m_line_vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(line_pts), line_pts);

        glUseProgram(m_line_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_line_shader, "u_ortho"),
                           1, GL_FALSE, ortho);
        glBindVertexArray(m_line_vao);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

    // ── Draw persistence scatter points ───────────────────────────────────────
    if (m_num_points > 0) {
        glUseProgram(m_shader);
        glUniformMatrix4fv(glGetUniformLocation(m_shader, "u_ortho"),
                           1, GL_FALSE, ortho);
        glUniform1f(glGetUniformLocation(m_shader, "u_min_persistence"),
                    m_min_persistence);
        glUniform1i(glGetUniformLocation(m_shader, "u_selected_dim"), -1);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_POINTS, 0, m_num_points);
        glBindVertexArray(0);
    }

    glUseProgram(0);
}

const tda::PersistencePoint* DiagramRenderer::hitTest(float, float) const {
    return nullptr;
}

} // namespace mmviz::render
