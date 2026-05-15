#pragma once
#include <vector>
#include <array>
#include "nn/ModelInference.h"

namespace mmviz::xai {

// Gradient-weighted Class Activation Mapping for 3D volumes
// Produces a saliency map S ∈ R^{X×Y×Z} highlighting voxels
// most responsible for the predicted class
class GradCAM {
public:
    struct SaliencyMap {
        std::vector<float>  values;      // normalized [0,1] importance per voxel
        std::array<int,3>   shape;       // {X, Y, Z}
        int                 target_class;

        float at(int x, int y, int z) const;
    };

    explicit GradCAM(const std::string& layer_name);

    SaliencyMap compute(const nn::ModelInference& model,
                        const std::vector<float>& input,
                        int target_class = -1) const;  // -1 = use predicted class

private:
    std::string m_layer_name;
};

} // namespace mmviz::xai
