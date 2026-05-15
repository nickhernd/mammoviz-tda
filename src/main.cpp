#include "io/DicomLoader.h"
#include "tda/PointCloud.h"
#include "tda/VietorisRips.h"
#include "nn/ModelInference.h"
#include "nn/FeatureExtractor.h"
#include "xai/GradCAM.h"
#include "xai/ManifoldProjector.h"
#include "render/RenderEngine.h"
#include "render/VolumeRenderer.h"
#include "render/DiagramRenderer.h"
#include "render/ManifoldRenderer.h"
#include "utils/Logger.h"
#include "utils/Config.h"
#include <memory>
#include <string>

int main(int argc, char** argv) {
    using namespace mmviz;

    utils::Config::instance().load("config.toml");
    utils::Logger::instance().setLevel(utils::LogLevel::INFO);

    // ── 1. Load medical volume ────────────────────────────────────────────────
    const std::string dicom_dir = (argc > 1) ? argv[1] : "data/samples/case001";
    LOG_INFO("Loading DICOM from: {}", dicom_dir);

    io::DicomLoader loader;
    auto volume = loader.load(dicom_dir);
    LOG_INFO("Volume loaded: {}x{}x{}x{}", volume->shape[0], volume->shape[1],
             volume->shape[2], volume->shape[3]);

    // ── 2. TDA: extract point cloud and compute persistence diagram ───────────
    LOG_INFO("Extracting point cloud...");
    auto cloud = tda::PointCloud::fromVolume(*volume);
    LOG_INFO("Point cloud: {} points", cloud.size());

    tda::VietorisRips rips;
    LOG_INFO("Computing Vietoris-Rips filtration...");
    auto diagram = rips.compute(cloud);
    LOG_INFO("Persistence diagram: {} points", diagram.points.size());

    auto tda_features = diagram.toFeatureVector();

    // ── 3. CNN inference with TDA features ───────────────────────────────────
    nn::ModelInference model("data/models/breast_cnn.onnx");
    if (!model.isLoaded()) {
        LOG_WARN("Model not found — running in visualization-only mode");
    }

    nn::FeatureExtractor extractor;
    auto image_features = extractor.extract(*volume);

    auto result = model.run(image_features, tda_features);
    LOG_INFO("Prediction: class={} confidence={:.2f}", result.predicted_class, result.confidence);
    if (result.predicted_class == 1)
        LOG_INFO("RESULT: MALIGNANT ({:.1f}% confidence)", result.confidence * 100.0f);
    else
        LOG_INFO("RESULT: BENIGN ({:.1f}% confidence)", result.confidence * 100.0f);

    // ── 4. XAI: volumetric GradCAM saliency ──────────────────────────────────
    xai::GradCAM gradcam("layer4");
    auto saliency = gradcam.computeVolumetric(model, extractor, *volume, tda_features);

    xai::ManifoldProjector projector;

    // ── 5. Build AppState for UI sidebar ─────────────────────────────────────
    render::AppState state;
    state.pred_class      = result.predicted_class;
    state.confidence      = result.confidence;
    state.logit_benign    = result.logits.size() > 0 ? result.logits[0] : 0.0f;
    state.logit_malignant = result.logits.size() > 1 ? result.logits[1] : 0.0f;

    // Count H0/H1/H2 points and max persistence from diagram
    float max_pers = 0.0f;
    for (const auto& pt : diagram.points) {
        float p = pt.persistence();
        if (p > max_pers) max_pers = p;
        if      (pt.dimension == 0) ++state.tda_h0;
        else if (pt.dimension == 1) ++state.tda_h1;
        else if (pt.dimension == 2) ++state.tda_h2;
    }
    state.tda_total_points    = (int)diagram.points.size();
    state.tda_max_persistence = max_pers;

    state.vol_x      = volume->shape[0];
    state.vol_y      = volume->shape[1];
    state.vol_z      = volume->shape[2];
    state.spacing_xy = volume->spacing[0];
    state.spacing_z  = volume->spacing[2];

    // Extract case ID from path
    {
        std::string path = dicom_dir;
        auto pos = path.find_last_of("/\\");
        state.case_id = (pos != std::string::npos) ? path.substr(pos+1) : path;
    }

    // ── 6. Render (GL context created inside engine.init) ────────────────────
    auto vol_renderer  = std::make_shared<render::VolumeRenderer>();
    auto diag_renderer = std::make_shared<render::DiagramRenderer>();
    auto mfld_renderer = std::make_shared<render::ManifoldRenderer>();

    render::RenderEngine engine{render::RenderConfig{}};
    engine.setVolumeRenderer(vol_renderer);
    engine.setDiagramRenderer(diag_renderer);
    engine.setManifoldRenderer(mfld_renderer);
    engine.setAppState(state);

    engine.onDiagramClick([&](float birth, float death, int dim) {
        LOG_INFO("Persistence point selected: birth={:.3f} death={:.3f} dim={}", birth, death, dim);
        auto voxels = vol_renderer->voxelsForPersistenceRegion(birth, death, dim);
        LOG_INFO("Highlighting {} voxels", voxels.size());
    });

    // GL context created here — all GL calls must come after this point
    engine.init();

    vol_renderer->uploadVolume(*volume);
    vol_renderer->uploadSaliency(saliency);
    vol_renderer->setRenderMode(render::VolumeRenderer::RenderMode::GradCAMOverlay);

    diag_renderer->setDiagram(diagram);

    engine.run();
    engine.shutdown();

    return 0;
}
