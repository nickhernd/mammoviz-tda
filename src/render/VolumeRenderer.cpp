#include "render/VolumeRenderer.h"
#include "utils/Logger.h"
#include <GL/glew.h>
#include <array>

namespace mmviz::render {

// ── Transfer function presets ─────────────────────────────────────────────────

VolumeRenderer::TransferFunction VolumeRenderer::TransferFunction::breastDefault() {
    TransferFunction tf;
    tf.lut.resize(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        // Dark background fades to white tissue, high intensities are yellow
        float a = (t < 0.1f) ? 0.0f : (t < 0.5f) ? (t - 0.1f) * 0.5f : t * 0.8f;
        tf.lut[i] = { t, t * 0.9f, t * 0.7f, a };
    }
    return tf;
}

VolumeRenderer::TransferFunction VolumeRenderer::TransferFunction::calcificationHighlight() {
    TransferFunction tf;
    tf.lut.resize(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float a = (t < 0.8f) ? 0.01f : (t - 0.8f) * 4.0f;  // only bright spots visible
        float r = (t > 0.8f) ? 1.0f : t * 0.3f;
        tf.lut[i] = { r, t * 0.3f, t * 0.3f, a };
    }
    return tf;
}

// ── VolumeRenderer stub (full shader-based implementation in Phase 2) ─────────

VolumeRenderer::VolumeRenderer() {
    m_tf = TransferFunction::breastDefault();
}

VolumeRenderer::~VolumeRenderer() {
    if (m_vol_tex) glDeleteTextures(1, &m_vol_tex);
    if (m_sal_tex) glDeleteTextures(1, &m_sal_tex);
}

void VolumeRenderer::uploadVolume(const io::VolumeData& vol) {
    if (m_vol_tex) glDeleteTextures(1, &m_vol_tex);
    glGenTextures(1, &m_vol_tex);
    glBindTexture(GL_TEXTURE_3D, m_vol_tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                 vol.shape[0], vol.shape[1], vol.shape[2],
                 0, GL_RED, GL_FLOAT, vol.data.data());
    LOG_INFO("Volume uploaded to GPU: {}×{}×{}", vol.shape[0], vol.shape[1], vol.shape[2]);
}

void VolumeRenderer::uploadSaliency(const xai::GradCAM::SaliencyMap& map) {
    if (map.values.empty()) return;
    if (m_sal_tex) glDeleteTextures(1, &m_sal_tex);
    glGenTextures(1, &m_sal_tex);
    glBindTexture(GL_TEXTURE_3D, m_sal_tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                 map.shape[0], map.shape[1], map.shape[2],
                 0, GL_RED, GL_FLOAT, map.values.data());
}

void VolumeRenderer::setRenderMode(RenderMode mode) { m_mode = mode; }
void VolumeRenderer::setTransferFunction(const TransferFunction& tf) { m_tf = tf; }

void VolumeRenderer::render(const float* /*view*/, const float* /*proj*/) {
    // Full ray-casting shader pipeline implemented in Phase 2
    // For now: verify GPU resources are bound
    if (m_vol_tex) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, m_vol_tex);
    }
}

std::vector<std::array<int,3>> VolumeRenderer::voxelsForPersistenceRegion(
    float /*birth*/, float /*death*/, int /*dimension*/) const
{
    // Stub: mapping persistence → voxel cluster implemented in Phase 2
    // Requires storing the Vietoris-Rips simplex→voxel lookup table
    return {};
}

} // namespace mmviz::render
