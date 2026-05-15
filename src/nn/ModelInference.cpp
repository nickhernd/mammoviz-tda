#include "nn/ModelInference.h"
#include "utils/Logger.h"

#ifndef NO_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace mmviz::nn {

struct ModelInference::Impl {
#ifndef NO_ONNX
    Ort::Env            env;
    Ort::SessionOptions opts;   // must be before session (init order = declaration order)
    Ort::Session        session;

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
    // Two separate input tensors matching the ONNX graph signature
    std::vector<float> img_buf(image_input);
    std::vector<float> tda_buf = tda_features.empty()
        ? std::vector<float>(192, 0.0f) : tda_features;

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> img_shape = { 1, (int64_t)img_buf.size() };
    std::array<int64_t, 2> tda_shape = { 1, (int64_t)tda_buf.size() };

    Ort::Value tensors[2] = {
        Ort::Value::CreateTensor<float>(mem, img_buf.data(), img_buf.size(),
                                        img_shape.data(), 2),
        Ort::Value::CreateTensor<float>(mem, tda_buf.data(), tda_buf.size(),
                                        tda_shape.data(), 2),
    };

    const char* input_names[]  = { "image_features", "tda_features" };
    const char* output_names[] = { "logits" };

    auto outputs = m_session->session.Run(
        Ort::RunOptions{nullptr}, input_names, tensors, 2, output_names, 1);

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
