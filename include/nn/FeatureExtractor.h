#pragma once
#include <vector>
#include "io/VolumeData.h"

namespace mmviz::nn {

struct FeatureExtractorParams {
    int  patch_size = 4;   // 4³=64 voxels; 8 patches → exactly 512 features
    int  n_patches  = 8;
    bool augment    = false;
};

class FeatureExtractor {
public:
    explicit FeatureExtractor(const FeatureExtractorParams& p = FeatureExtractorParams{});
    std::vector<float> extract(const io::VolumeData& vol) const;

private:
    FeatureExtractorParams m_params;
};

} // namespace mmviz::nn
