#pragma once
#include <string>
#include <memory>
#include "io/VolumeData.h"

namespace mmviz::io {

struct DicomLoadOptions {
    bool  normalize_intensity = true;
    bool  load_temporal       = true;
    float intensity_threshold = 0.85f;
};

// Loads DICOM series (DCE-MRI or tomosynthesis) into a 4D tensor
class DicomLoader {
public:
    explicit DicomLoader(const DicomLoadOptions& opts = DicomLoadOptions{});

    std::unique_ptr<VolumeData> load(const std::string& dicom_dir) const;
    std::unique_ptr<VolumeData> loadMHA(const std::string& mha_path) const;

private:
    DicomLoadOptions m_opts;
};

} // namespace mmviz::io
