#pragma once
#include <string>
#include <vector>
#include <memory>

namespace mmviz::nn {

// Wraps ONNX Runtime for CNN/ViT model inference
// Models are trained in Python/PyTorch and exported to .onnx
class ModelInference {
public:
    struct InferenceResult {
        std::vector<float> logits;       // raw class scores
        int                predicted_class;
        float              confidence;
        // Activations from intermediate layers (for XAI)
        std::vector<std::vector<float>> layer_activations;
    };

    explicit ModelInference(const std::string& onnx_model_path);
    ~ModelInference();

    // Run inference on a flattened image tensor + TDA feature vector
    // Input: concatenated [image_features | tda_features]
    InferenceResult run(const std::vector<float>& image_input,
                        const std::vector<float>& tda_features) const;

    // Names of intermediate layers to capture for XAI
    void setHookLayers(const std::vector<std::string>& layer_names);

    bool isLoaded() const { return m_session != nullptr; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_session;
    std::vector<std::string> m_hook_layers;
};

} // namespace mmviz::nn
