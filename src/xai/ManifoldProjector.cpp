#include "xai/ManifoldProjector.h"
#include "utils/Logger.h"
#include <cmath>
#include <algorithm>

namespace mmviz::xai {

ManifoldProjector::ManifoldProjector(const ManifoldParams& params) : m_params(params) {}

static Eigen::MatrixXf centerAndScale(const Eigen::MatrixXf& X) {
    Eigen::VectorXf mean = X.colwise().mean();
    Eigen::MatrixXf Xc   = X.rowwise() - mean.transpose();
    float scale = Xc.norm();
    return (scale > 0.0f) ? Xc / scale : Xc;
}

void ManifoldProjector::fit(const Eigen::MatrixXf& activations,
                            const std::vector<int>& /*labels*/)
{
    LOG_INFO("Fitting manifold projector: {} samples × {} dims",
             activations.rows(), activations.cols());

    Eigen::MatrixXf Xc = centerAndScale(activations);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);

    int k = std::min(3, (int)svd.singularValues().size());
    m_embedding = svd.matrixU().leftCols(k) *
                  svd.singularValues().head(k).asDiagonal();

    for (int j = 0; j < m_embedding.cols(); ++j) {
        float mx = m_embedding.col(j).cwiseAbs().maxCoeff();
        if (mx > 0.0f) m_embedding.col(j) /= mx;
    }

    m_fitted = true;
    LOG_INFO("Manifold fit complete (PCA baseline)");
}

std::vector<ProjectedPoint> ManifoldProjector::transform(
    const Eigen::MatrixXf& /*activations*/,
    const std::vector<std::string>& ids) const
{
    if (!m_fitted) {
        LOG_ERROR("ManifoldProjector not fitted");
        return {};
    }

    std::vector<ProjectedPoint> result;
    result.reserve(m_embedding.rows());

    for (int i = 0; i < m_embedding.rows(); ++i) {
        ProjectedPoint pt;
        pt.coords     = { m_embedding(i, 0),
                          (m_embedding.cols() > 1) ? m_embedding(i, 1) : 0.0f,
                          (m_embedding.cols() > 2) ? m_embedding(i, 2) : 0.0f };
        pt.label      = 0;
        pt.confidence = 1.0f;
        pt.sample_id  = (i < (int)ids.size()) ? ids[i] : std::to_string(i);
        result.push_back(pt);
    }

    return result;
}

} // namespace mmviz::xai
