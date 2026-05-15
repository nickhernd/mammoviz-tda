#pragma once
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "io/VolumeData.h"

namespace mmviz::io {

// Loads DICOM series (DCE-MRI or tomosynthesis) into a 4D tensor
// T ∈ R^{X×Y×Z×t} where t=1 for static volumes (DBT), t>1 for DCE-MRI
class DicomLoader {
public:
    struct LoadOptions {
        bool normalize_intensity = true;   // map HU to [0,1]
        bool load_temporal       = true;   // for DCE-MRI: load all phases
        float intensity_threshold = 0.85f; // for microcalcification extraction
    };

    explicit DicomLoader(const LoadOptions& opts = {});

    // Load a DICOM series from a directory
    std::unique_ptr<VolumeData> load(const std::string& dicom_dir) const;

    // Load a single .mhd/.mha file (common in public datasets like CBIS-DDSM)
    std::unique_ptr<VolumeData> loadMHA(const std::string& mha_path) const;

private:
    LoadOptions m_opts;
};

} // namespace mmviz::io
