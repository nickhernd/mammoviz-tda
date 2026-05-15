#pragma once
#include <vector>
#include <array>
#include "io/VolumeData.h"

namespace mmviz::tda {

using Point3D = std::array<float, 3>;

// Extracts a 3D point cloud P ⊂ R³ from a segmented medical volume
// Points are voxel coordinates (x,y,z) of structures above intensity threshold
class PointCloud {
public:
    struct ExtractionParams {
        float  intensity_min = 0.85f;  // min normalized intensity to include
        float  intensity_max = 1.00f;
        int    max_points    = 50000;  // subsample if larger (for performance)
        bool   apply_spacing = true;   // scale coords by voxel spacing (→ mm)
    };

    static PointCloud fromVolume(const io::VolumeData& vol,
                                 const ExtractionParams& params = {});

    const std::vector<Point3D>& points() const { return m_points; }
    size_t size() const { return m_points.size(); }

    // Writes to .off or .csv for external tools
    void exportOFF(const std::string& path) const;

private:
    std::vector<Point3D> m_points;
};

} // namespace mmviz::tda
