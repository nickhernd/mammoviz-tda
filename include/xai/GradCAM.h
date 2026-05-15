#pragma once
#include <vector>
#include <array>
#include "nn/ModelInference.h"
#include "nn/FeatureExtractor.h"
#include "io/VolumeData.h"

namespace mmviz::xai {

class GradCAM {
public:
    struct SaliencyMap {
        std::vector<float>  values;      // normalized [0,1] importance per voxel
        std::array<int,3>   shape;       // {X, Y, Z}
        int                 target_class;

        float at(int x, int y, int z) const;
    };

    explicit GradCAM(const std::string& layer_name);

    // Volumetric perturbation saliency: perturbs 3D spatial blocks of the volume,
    // re-extracts features, re-runs inference, and measures confidence change.
    // Produces a proper 3D saliency map matching the volume dimensions.
    // grid_n = number of blocks per axis (grid_n^3 total perturbations).
    SaliencyMap computeVolumetric(const nn::ModelInference& model,
                                  const nn::FeatureExtractor& extractor,
                                  const io::VolumeData& vol,
                                  const std::vector<float>& tda_features = {},
                                  int grid_n = 4,
                                  int target_class = -1) const;

    // Legacy feature-space perturbation (kept for fallback)
    SaliencyMap compute(const nn::ModelInference& model,
                        const std::vector<float>& input,
                        const std::vector<float>& tda_features = {},
                        int target_class = -1) const;

private:
    std::string m_layer_name;
};

} // namespace mmviz::xai
