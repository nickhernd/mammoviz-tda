#include "render/DiagramRenderer.h"
#include "utils/Logger.h"
#include <GL/glew.h>
#include <cmath>
#include <algorithm>

namespace mmviz::render {

DiagramRenderer::DiagramRenderer() {}
DiagramRenderer::~DiagramRenderer() {
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

void DiagramRenderer::setDiagram(const tda::PersistenceDiagram& diagram) {
    m_current_diagram = diagram;
    LOG_INFO("Diagram set: {} persistence points", diagram.points.size());
}

void DiagramRenderer::setTemporalDiagrams(const std::vector<tda::PersistenceDiagram>& diags) {
    m_temporal_diags = diags;
}

void DiagramRenderer::setCurrentPhase(int phase_index) {
    m_phase = phase_index;
    if (phase_index < (int)m_temporal_diags.size())
        m_current_diagram = m_temporal_diags[phase_index];
}

void DiagramRenderer::setMinPersistence(float min_p) {
    m_min_persistence = min_p;
}

void DiagramRenderer::render(int /*px*/, int /*py*/, int /*pw*/, int /*ph*/) {
    // Full scatter-plot shader pipeline implemented in Phase 2 (diagram.vert/frag)
    // Points are rendered as circles with color = dimension, size ∝ persistence
    // The diagonal line y=x separates real features from noise
}

const tda::PersistencePoint* DiagramRenderer::hitTest(float sx, float sy) const {
    // Find the persistence point whose diagram coordinates are nearest to screen pos
    // Requires storing the screen→diagram coordinate mapping set during render()
    return nullptr;
}

} // namespace mmviz::render
