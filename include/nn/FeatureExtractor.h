#pragma once
#include <vector>
#include "io/VolumeData.h"

namespace mmviz::nn {

// Extracts a flat feature vector from a VolumeData for CNN input
// Applies patch sampling + normalization to produce a fixed-size descriptor
class FeatureExtractor {
public:
    struct Params {
        int   patch_size  = 64;    // cubic patch side length in voxels
        int   n_patches   = 16;    // number of random patches to sample
        bool  augment     = false; // random flips/rotations (training only)
    };

    explicit FeatureExtractor(const Params& p = {});

    // Returns flattened patch tensor for CNN input
    std::vector<float> extract(const io::VolumeData& vol) const;

private:
    Params m_params;
};

} // namespace mmviz::nn
