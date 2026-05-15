#include "xai/ManifoldProjector.h"
#include "utils/Logger.h"
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace mmviz::xai {

ManifoldProjector::ManifoldProjector(const Params& params) : m_params(params) {}

// ── PCA-based 3D projection (baseline; replace with UMAP for publication) ────
// A proper UMAP implementation in C++ requires either:
//   (a) umappp library (https://github.com/LTLA/umappp)
//   (b) Python bridge for sklearn.manifold.UMAP via pybind11
//   (c) Export UMAP embedding from Python and load here
//
// For initial development we ship a PCA baseline that is already interactive.

static Eigen::MatrixXf centerAndScale(const Eigen::MatrixXf& X) {
    Eigen::VectorXf mean = X.colwise().mean();
    Eigen::MatrixXf Xc   = X.rowwise() - mean.transpose();
    float scale = Xc.norm();
    return (scale > 0.0f) ? Xc / scale : Xc;
}

void ManifoldProjector::fit(const Eigen::MatrixXf& activations,
                            const std::vector<int>& labels)
{
    LOG_INFO("Fitting manifold projector: {} samples × {} dims",
             activations.rows(), activations.cols());

    Eigen::MatrixXf Xc = centerAndScale(activations);

    // Covariance matrix (economy SVD for large D)
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Take the first 3 principal components
    int k = std::min(3, (int)svd.singularValues().size());
    m_embedding = svd.matrixU().leftCols(k) *
                  svd.singularValues().head(k).asDiagonal();

    // Normalize embedding to [-1, 1]³ for rendering
    for (int j = 0; j < m_embedding.cols(); ++j) {
        float mx = m_embedding.col(j).cwiseAbs().maxCoeff();
        if (mx > 0.0f) m_embedding.col(j) /= mx;
    }

    m_fitted = true;
    LOG_INFO("Manifold fit complete ({} retained)", m_params.method == Method::PCA ? "PCA" : "PCA-approx");
}

std::vector<ManifoldProjector::ProjectedPoint> ManifoldProjector::transform(
    const Eigen::MatrixXf& activations,
    const std::vector<std::string>& ids) const
{
    if (!m_fitted) {
        LOG_ERROR("ManifoldProjector not fitted — call fit() first");
        return {};
    }

    std::vector<ProjectedPoint> result;
    result.reserve(activations.rows());

    for (int i = 0; i < activations.rows(); ++i) {
        ProjectedPoint pt;
        pt.coords      = { (i < m_embedding.rows()) ? m_embedding(i, 0) : 0.0f,
                           (i < m_embedding.rows() && m_embedding.cols() > 1) ? m_embedding(i, 1) : 0.0f,
                           (i < m_embedding.rows() && m_embedding.cols() > 2) ? m_embedding(i, 2) : 0.0f };
        pt.label       = 0;
        pt.confidence  = 1.0f;
        pt.sample_id   = (i < (int)ids.size()) ? ids[i] : std::to_string(i);
        result.push_back(pt);
    }

    return result;
}

} // namespace mmviz::xai
