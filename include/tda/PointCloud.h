#pragma once
#include <vector>
#include <array>
#include <string>
#include "io/VolumeData.h"

namespace mmviz::tda {

using Point3D = std::array<float, 3>;

struct PointCloudParams {
    float intensity_min = 0.85f;
    float intensity_max = 1.00f;
    int   max_points    = 50000;
    bool  apply_spacing = true;
};

// Extracts a 3D point cloud P ⊂ R³ from a segmented medical volume
class PointCloud {
public:
    static PointCloud fromVolume(const io::VolumeData& vol,
                                 const PointCloudParams& params = PointCloudParams{});

    const std::vector<Point3D>& points() const { return m_points; }
    size_t size() const { return m_points.size(); }

    void exportOFF(const std::string& path) const;

private:
    std::vector<Point3D> m_points;
};

} // namespace mmviz::tda
