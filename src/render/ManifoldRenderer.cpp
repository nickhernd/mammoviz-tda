#include "render/ManifoldRenderer.h"
#include "utils/Logger.h"
#include <GL/glew.h>

namespace mmviz::render {

ManifoldRenderer::ManifoldRenderer() {}
ManifoldRenderer::~ManifoldRenderer() {
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
}

void ManifoldRenderer::setPoints(const std::vector<xai::ProjectedPoint>& pts) {
    m_points = pts;
    LOG_INFO("Manifold: {} projected samples loaded", pts.size());
}

void ManifoldRenderer::setSelectedSample(const std::string& id) {
    m_selected_id = id;
}

void ManifoldRenderer::showTrajectories(bool enabled) {
    m_trajectories = enabled;
}

void ManifoldRenderer::render(int /*px*/, int /*py*/, int /*pw*/, int /*ph*/,
                               const float* /*view*/, const float* /*proj*/) {
    // Full 3D scatter implementation using manifold.vert/frag in Phase 2
}

std::string ManifoldRenderer::hitTest(float, float, float, float, float, float) const {
    return {};
}

} // namespace mmviz::render
