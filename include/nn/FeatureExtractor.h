#pragma once
#include <vector>
#include "io/VolumeData.h"

namespace mmviz::nn {

struct FeatureExtractorParams {
    int  patch_size = 64;
    int  n_patches  = 16;
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
