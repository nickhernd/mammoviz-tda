#pragma once
#include <vector>
#include <memory>
#include <string>
#include "xai/ManifoldProjector.h"

namespace mmviz::render {

// Renders the 3D latent space manifold of the CNN
// Each point = one patient/sample; position = UMAP projection of latent activations
// Color = predicted class confidence; shape = ground truth label
// Allows clinicians to see which cases cluster together in the model's "mind"
class ManifoldRenderer {
public:
    ManifoldRenderer();
    ~ManifoldRenderer();

    void setPoints(const std::vector<xai::ProjectedPoint>& pts);

    // Highlight the currently selected sample
    void setSelectedSample(const std::string& sample_id);

    // Toggle trajectory lines connecting temporal phases of the same patient
    void showTrajectories(bool enabled);

    void render(int panel_x, int panel_y, int panel_w, int panel_h,
                const float* view_matrix, const float* proj_matrix);

    // Returns sample_id of the point closest to a screen ray
    std::string hitTest(float ray_ox, float ray_oy, float ray_oz,
                        float ray_dx, float ray_dy, float ray_dz) const;

private:
    unsigned int m_shader     = 0;
    unsigned int m_vao        = 0;
    unsigned int m_vbo        = 0;

    std::vector<xai::ProjectedPoint> m_points;
    std::string m_selected_id;
    bool        m_trajectories = false;
};

} // namespace mmviz::render
