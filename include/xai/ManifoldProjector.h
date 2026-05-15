#pragma once
#include <vector>
#include <array>
#include <string>
#include <Eigen/Dense>

namespace mmviz::xai {

enum class ManifoldMethod { UMAP, TSNE, PCA };

struct ManifoldParams {
    ManifoldMethod method      = ManifoldMethod::UMAP;
    int            n_neighbors = 15;
    float          min_dist    = 0.1f;
    int            n_epochs    = 200;
    int            random_seed = 42;
};

struct ProjectedPoint {
    std::array<float, 3> coords;
    int                  label;
    float                confidence;
    std::string          sample_id;
};

// Projects high-dimensional CNN latent space to R³ for interactive visualization
class ManifoldProjector {
public:
    explicit ManifoldProjector(const ManifoldParams& params = ManifoldParams{});

    void fit(const Eigen::MatrixXf& activations, const std::vector<int>& labels);

    std::vector<ProjectedPoint> transform(
        const Eigen::MatrixXf& activations,
        const std::vector<std::string>& ids = std::vector<std::string>{}) const;

    bool isFitted() const { return m_fitted; }

private:
    ManifoldParams  m_params;
    bool            m_fitted = false;
    Eigen::MatrixXf m_embedding;
};

} // namespace mmviz::xai
