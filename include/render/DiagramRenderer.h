#pragma once
#include <memory>
#include "tda/PersistenceDiagram.h"
#include "render/RenderEngine.h"

namespace mmviz::render {

// Renders an interactive persistence diagram in a 2D panel
// X-axis = birth ε, Y-axis = death ε
// Points above the diagonal represent real topological features
// Color-coded by dimension: β₀ (blue), β₁ (green), β₂ (red)
class DiagramRenderer {
public:
    DiagramRenderer();
    ~DiagramRenderer();

    void setDiagram(const tda::PersistenceDiagram& diagram);

    // For DCE-MRI: animate through temporal diagrams
    void setTemporalDiagrams(const std::vector<tda::PersistenceDiagram>& diags);
    void setCurrentPhase(int phase_index);

    // Filter to show only features above this persistence threshold
    void setMinPersistence(float min_p);

    void render(int panel_x, int panel_y, int panel_w, int panel_h);

    // Returns the persistence point closest to screen coordinates (for click handling)
    const tda::PersistencePoint* hitTest(float screen_x, float screen_y) const;

private:
    unsigned int m_shader      = 0;
    unsigned int m_vao         = 0;
    unsigned int m_vbo         = 0;
    unsigned int m_line_shader = 0;
    unsigned int m_line_vao    = 0;
    unsigned int m_line_vbo    = 0;
    int          m_num_points  = 0;
    float        m_scale_max   = 1.0f;
    bool         m_shaders_ok  = false;
    bool         m_pts_dirty   = true;

    tda::PersistenceDiagram              m_current_diagram;
    std::vector<tda::PersistenceDiagram> m_temporal_diags;
    float m_min_persistence = 0.0f;
    int   m_phase           = 0;

    void initShaders();
    void uploadPoints();
};

} // namespace mmviz::render
