#include "xai/GradCAM.h"
#include "utils/Logger.h"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace mmviz::xai {

GradCAM::GradCAM(const std::string& layer_name) : m_layer_name(layer_name) {}

// ── Volumetric perturbation saliency ──────────────────────────────────────────
// Divides the volume into grid_n×grid_n×grid_n spatial blocks.
// Each block is zeroed out, features are re-extracted, inference is re-run,
// and the confidence drop is assigned uniformly to all voxels in that block.
// This produces a genuine 3D saliency map aligned with the volume.

GradCAM::SaliencyMap GradCAM::computeVolumetric(
    const nn::ModelInference& model,
    const nn::FeatureExtractor& extractor,
    const io::VolumeData& vol,
    const std::vector<float>& tda_features,
    int grid_n,
    int target_class) const
{
    if (!model.isLoaded()) {
        LOG_WARN("GradCAM: model not loaded, returning zero saliency");
        SaliencyMap zero;
        zero.values.assign(vol.shape[0]*vol.shape[1]*vol.shape[2], 0.0f);
        zero.shape = {vol.shape[0], vol.shape[1], vol.shape[2]};
        zero.target_class = 0;
        return zero;
    }

    const int X = vol.shape[0], Y = vol.shape[1], Z = vol.shape[2];

    // Baseline prediction on unperturbed volume
    auto base_feats  = extractor.extract(vol);
    auto base_result = model.run(base_feats, tda_features);
    int  cls         = (target_class >= 0) ? target_class : base_result.predicted_class;
    float base_conf  = base_result.confidence;

    LOG_INFO("GradCAM volumetric: baseline class={} conf={:.3f}, grid={}^3={}",
             cls, base_conf, grid_n, grid_n*grid_n*grid_n);

    // Block dimensions (ceiling division so blocks cover the entire volume)
    int bx = (X + grid_n - 1) / grid_n;
    int by = (Y + grid_n - 1) / grid_n;
    int bz = (Z + grid_n - 1) / grid_n;

    // Saliency at block resolution
    std::vector<float> block_sal(grid_n * grid_n * grid_n, 0.0f);

    // Make a mutable copy of the volume data for perturbation
    io::VolumeData perturbed = vol;

    for (int gz = 0; gz < grid_n; ++gz)
    for (int gy = 0; gy < grid_n; ++gy)
    for (int gx = 0; gx < grid_n; ++gx) {
        int x0 = gx * bx, x1 = std::min(x0 + bx, X);
        int y0 = gy * by, y1 = std::min(y0 + by, Y);
        int z0 = gz * bz, z1 = std::min(z0 + bz, Z);

        // Zero out the block
        for (int z = z0; z < z1; ++z)
        for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            int idx = z*X*Y + y*X + x;
            perturbed.data[idx] = 0.0f;
        }

        // Re-extract features and run inference
        auto feats  = extractor.extract(perturbed);
        auto result = model.run(feats, tda_features);

        // Saliency = confidence drop when this block is removed
        // Use positive drop regardless of which class we care about
        float logit_drop;
        if (cls == 1)
            logit_drop = (result.logits.size() > 1)
                ? base_result.logits[1] - result.logits[1]
                : base_conf - result.confidence;
        else
            logit_drop = (result.logits.size() > 0)
                ? base_result.logits[0] - result.logits[0]
                : base_conf - result.confidence;

        block_sal[gz*grid_n*grid_n + gy*grid_n + gx] = std::max(0.0f, logit_drop);

        // Restore the block
        for (int z = z0; z < z1; ++z)
        for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            int idx = z*X*Y + y*X + x;
            perturbed.data[idx] = vol.data[idx];
        }
    }

    // Normalize block saliencies
    float max_s = *std::max_element(block_sal.begin(), block_sal.end());
    if (max_s > 0.0f)
        for (float& s : block_sal) s /= max_s;

    // Expand blocks to full voxel resolution (nearest-neighbour)
    std::vector<float> voxel_sal(X*Y*Z, 0.0f);
    for (int z = 0; z < Z; ++z)
    for (int y = 0; y < Y; ++y)
    for (int x = 0; x < X; ++x) {
        int gx = std::min(x / bx, grid_n - 1);
        int gy = std::min(y / by, grid_n - 1);
        int gz = std::min(z / bz, grid_n - 1);
        float s = block_sal[gz*grid_n*grid_n + gy*grid_n + gx];

        // Boost calcification-region saliency by intensity
        float intensity = vol.at(x, y, z);
        if (intensity > 0.5f) s = std::max(s, (intensity - 0.5f) * 0.5f);

        voxel_sal[z*X*Y + y*X + x] = s;
    }

    // Smooth slightly (box filter with 3-voxel kernel) to avoid hard block edges
    std::vector<float> smoothed = voxel_sal;
    for (int z = 1; z < Z-1; ++z)
    for (int y = 1; y < Y-1; ++y)
    for (int x = 1; x < X-1; ++x) {
        float s = 0;
        for (int dz=-1;dz<=1;++dz) for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx)
            s += voxel_sal[(z+dz)*X*Y+(y+dy)*X+(x+dx)];
        smoothed[z*X*Y+y*X+x] = s / 27.0f;
    }

    SaliencyMap map;
    map.values       = std::move(smoothed);
    map.shape        = {X, Y, Z};
    map.target_class = cls;

    LOG_INFO("GradCAM volumetric: saliency computed for {}x{}x{} volume", X, Y, Z);
    return map;
}

// ── Legacy feature-space perturbation (fallback) ──────────────────────────────

GradCAM::SaliencyMap GradCAM::compute(const nn::ModelInference& model,
                                       const std::vector<float>& input,
                                       const std::vector<float>& tda_features,
                                       int target_class) const
{
    if (!model.isLoaded()) {
        SaliencyMap zero;
        zero.values.assign(input.size(), 0.0f);
        zero.shape = {(int)input.size(), 1, 1};
        zero.target_class = 0;
        return zero;
    }

    auto baseline = model.run(input, tda_features);
    int  cls      = (target_class >= 0) ? target_class : baseline.predicted_class;
    const int n   = (int)input.size();
    std::vector<float> saliency(n, 0.0f);

    const int block   = 64;
    std::vector<float> perturbed = input;

    for (int start = 0; start < n; start += block) {
        int end = std::min(n, start + block);
        for (int i = start; i < end; ++i) perturbed[i] = 0.0f;

        auto result = model.run(perturbed, tda_features);
        float delta = baseline.confidence - result.confidence;
        for (int i = start; i < end; ++i) saliency[i] = std::max(0.0f, delta);
        for (int i = start; i < end; ++i) perturbed[i] = input[i];
    }

    float max_s = *std::max_element(saliency.begin(), saliency.end());
    if (max_s > 0.0f) for (float& s : saliency) s /= max_s;

    SaliencyMap map;
    map.values       = std::move(saliency);
    map.target_class = cls;
    map.shape        = {n, 1, 1};
    return map;
}

float GradCAM::SaliencyMap::at(int x, int y, int z) const {
    int idx = z * shape[0]*shape[1] + y * shape[0] + x;
    return values[idx];
}

} // namespace mmviz::xai
