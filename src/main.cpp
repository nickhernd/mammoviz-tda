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

    // ── 4. XAI: GradCAM saliency + manifold projection ───────────────────────
    xai::GradCAM gradcam("layer4");
    auto saliency = gradcam.compute(model, image_features);

    xai::ManifoldProjector projector;
    // (In practice: fit on dataset; here we just project the current sample)

    // ── 5. Render ─────────────────────────────────────────────────────────────
    auto vol_renderer  = std::make_shared<render::VolumeRenderer>();
    auto diag_renderer = std::make_shared<render::DiagramRenderer>();
    auto mfld_renderer = std::make_shared<render::ManifoldRenderer>();

    vol_renderer->uploadVolume(*volume);
    vol_renderer->uploadSaliency(saliency);
    vol_renderer->setRenderMode(render::VolumeRenderer::RenderMode::GradCAMOverlay);

    diag_renderer->setDiagram(diagram);

    render::RenderEngine engine;
    engine.setVolumeRenderer(vol_renderer);
    engine.setDiagramRenderer(diag_renderer);
    engine.setManifoldRenderer(mfld_renderer);

    // Linked selection: clicking a persistence point highlights tissue in volume
    engine.onDiagramClick([&](float birth, float death, int dim) {
        LOG_INFO("Persistence point selected: birth={:.3f} death={:.3f} dim={}", birth, death, dim);
        auto voxels = vol_renderer->voxelsForPersistenceRegion(birth, death, dim);
        LOG_INFO("Highlighting {} voxels", voxels.size());
    });

    engine.init();
    engine.run();
    engine.shutdown();

    return 0;
}
