#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>

namespace mmviz::xai {

// Projects high-dimensional CNN latent space to R³ for interactive visualization
// Implements topology-preserving dimensionality reduction (UMAP-like)
// Output: 3D embedding of the latent manifold the model uses to classify
class ManifoldProjector {
public:
    enum class Method {
        UMAP,   // topology-preserving (recommended)
        TSNE,   // local structure focused
        PCA,    // linear baseline
    };

    struct ProjectedPoint {
        std::array<float, 3> coords;  // 3D position in manifold space
        int                  label;   // ground truth class (0=benign, 1=malignant)
        float                confidence;
        std::string          sample_id;
    };

    struct Params {
        Method method        = Method::UMAP;
        int    n_neighbors   = 15;    // UMAP: local connectivity
        float  min_dist      = 0.1f;  // UMAP: minimum spread
        int    n_epochs      = 200;
        int    random_seed   = 42;
    };

    explicit ManifoldProjector(const Params& params = {});

    // Fit on a dataset of N high-dimensional activation vectors
    // activations: N × D matrix (N samples, D = latent dim)
    void fit(const Eigen::MatrixXf& activations, const std::vector<int>& labels);

    // Project new samples into the fitted embedding
    std::vector<ProjectedPoint> transform(const Eigen::MatrixXf& activations,
                                          const std::vector<std::string>& ids = {}) const;

    bool isFitted() const { return m_fitted; }

private:
    Params m_params;
    bool   m_fitted = false;
    Eigen::MatrixXf m_embedding;
};

} // namespace mmviz::xai
