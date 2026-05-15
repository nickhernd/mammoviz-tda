#include "nn/ModelInference.h"
#include "utils/Logger.h"

#ifndef NO_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace mmviz::nn {

struct ModelInference::Impl {
#ifndef NO_ONNX
    Ort::Env          env;
    Ort::Session      session;
    Ort::SessionOptions opts;

    Impl(const std::string& path)
        : env(ORT_LOGGING_LEVEL_WARNING, "mammoviz")
        , session(env, path.c_str(), opts)
    {}
#endif
};

ModelInference::ModelInference(const std::string& onnx_path) {
#ifndef NO_ONNX
    try {
        m_session = std::make_unique<Impl>(onnx_path);
        LOG_INFO("ONNX model loaded: {}", onnx_path);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX load failed: {}", e.what());
    }
#else
    LOG_WARN("Built without ONNX Runtime — inference disabled");
#endif
}

ModelInference::~ModelInference() = default;

ModelInference::InferenceResult ModelInference::run(
    const std::vector<float>& image_input,
    const std::vector<float>& tda_features) const
{
    InferenceResult result;

    if (!isLoaded()) {
        LOG_WARN("Model not loaded — returning dummy result");
        result.logits          = { 0.5f, 0.5f };
        result.predicted_class = 0;
        result.confidence      = 0.5f;
        return result;
    }

#ifndef NO_ONNX
    // Concatenate image features + TDA features into one input tensor
    std::vector<float> input;
    input.insert(input.end(), image_input.begin(),  image_input.end());
    input.insert(input.end(), tda_features.begin(), tda_features.end());

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape = { 1, (int64_t)input.size() };

    auto tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[]  = { "input" };
    const char* output_names[] = { "output" };

    auto outputs = m_session->session.Run(
        Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);

    float* logits = outputs[0].GetTensorMutableData<float>();
    result.logits = { logits[0], logits[1] };

    // Softmax
    float e0 = std::exp(logits[0]), e1 = std::exp(logits[1]);
    float sum = e0 + e1;
    result.predicted_class = (e1 / sum > 0.5f) ? 1 : 0;
    result.confidence      = std::max(e0, e1) / sum;
#endif

    return result;
}

void ModelInference::setHookLayers(const std::vector<std::string>& layer_names) {
    m_hook_layers = layer_names;
}

} // namespace mmviz::nn
