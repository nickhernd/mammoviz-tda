#include "xai/GradCAM.h"
#include "utils/Logger.h"
#include <numeric>
#include <cmath>

namespace mmviz::xai {

GradCAM::GradCAM(const std::string& layer_name) : m_layer_name(layer_name) {}

GradCAM::SaliencyMap GradCAM::compute(const nn::ModelInference& model,
                                       const std::vector<float>& input,
                                       int target_class) const
{
    // NOTE: True GradCAM requires backpropagation through the model.
    // ONNX Runtime does not expose gradients natively.
    // Production implementation should either:
    //   (a) Export model with gradient tape in LibTorch C++ API
    //   (b) Use ONNX Runtime Training API
    //   (c) Approximate with LIME/SHAP perturbation sampling
    //
    // Here we implement option (c): perturbation-based saliency
    // (drop patches of the input and measure prediction change)

    LOG_WARN("GradCAM: using perturbation-based approximation (ONNX mode)");

    auto baseline = model.run(input, {});
    int  cls      = (target_class >= 0) ? target_class : baseline.predicted_class;

    const int n = input.size();
    std::vector<float> saliency(n, 0.0f);

    // Perturb blocks of 1000 features at a time
    const int block = 1000;
    std::vector<float> perturbed = input;

    for (int start = 0; start < n; start += block) {
        int end = std::min(n, start + block);
        // Zero out this block
        for (int i = start; i < end; ++i) perturbed[i] = 0.0f;

        auto result = model.run(perturbed, {});
        float delta = baseline.confidence - result.confidence;

        // Assign saliency proportional to confidence drop
        for (int i = start; i < end; ++i) saliency[i] = std::max(0.0f, delta);

        // Restore
        for (int i = start; i < end; ++i) perturbed[i] = input[i];
    }

    // Normalize to [0,1]
    float max_s = *std::max_element(saliency.begin(), saliency.end());
    if (max_s > 0.0f)
        for (float& s : saliency) s /= max_s;

    // Shape: treat as a 3D volume (flat for now; caller knows the shape)
    SaliencyMap map;
    map.values       = std::move(saliency);
    map.target_class = cls;
    map.shape        = { n, 1, 1 };  // overridden by VolumeRenderer when uploading

    return map;
}

float GradCAM::SaliencyMap::at(int x, int y, int z) const {
    int idx = z * shape[0]*shape[1] + y * shape[0] + x;
    return values[idx];
}

} // namespace mmviz::xai
