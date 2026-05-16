#pragma once
#include <string>
#include <memory>
#include "io/VolumeData.h"

namespace mmviz::io {

struct DicomLoadOptions {
    bool  normalize_intensity = true;
    bool  load_temporal       = true;
    float intensity_threshold = 0.85f;
    int   max_dimension       = 256;  // downsample so no axis exceeds this (0 = no limit)
};

// Loads DICOM series (DCE-MRI or tomosynthesis) into a 4D tensor
class DicomLoader {
public:
    explicit DicomLoader(const DicomLoadOptions& opts = DicomLoadOptions{});

    std::unique_ptr<VolumeData> load(const std::string& dicom_dir) const;
    std::unique_ptr<VolumeData> loadMHA(const std::string& mha_path) const;

private:
    DicomLoadOptions m_opts;

    static void downsample(VolumeData& vol, int max_dim);
};

} // namespace mmviz::io
