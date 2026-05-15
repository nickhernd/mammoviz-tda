#include "nn/FeatureExtractor.h"
#include "utils/Logger.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace mmviz::nn {

FeatureExtractor::FeatureExtractor(const FeatureExtractorParams& p) : m_params(p) {}

std::vector<float> FeatureExtractor::extract(const io::VolumeData& vol) const {
    const int P = m_params.patch_size;
    const int X = vol.shape[0], Y = vol.shape[1], Z = vol.shape[2];

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dx(0, std::max(0, X - P));
    std::uniform_int_distribution<int> dy(0, std::max(0, Y - P));
    std::uniform_int_distribution<int> dz(0, std::max(0, Z - P));

    std::vector<float> features;
    features.reserve(m_params.n_patches * P * P * P);

    for (int i = 0; i < m_params.n_patches; ++i) {
        int ox = dx(rng), oy = dy(rng), oz = dz(rng);
        for (int z = 0; z < P && (oz+z) < Z; ++z)
        for (int y = 0; y < P && (oy+y) < Y; ++y)
        for (int x = 0; x < P && (ox+x) < X; ++x)
            features.push_back(vol.at(ox+x, oy+y, oz+z));
    }

    // Pad to fixed size if volume was smaller than patch
    features.resize(m_params.n_patches * P * P * P, 0.0f);

    LOG_INFO("Extracted {} feature values from {} patches", features.size(), m_params.n_patches);
    return features;
}

} // namespace mmviz::nn
