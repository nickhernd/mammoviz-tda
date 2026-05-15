#include "tda/PointCloud.h"
#include "utils/Logger.h"
#include <fstream>
#include <random>
#include <algorithm>

namespace mmviz::tda {

PointCloud PointCloud::fromVolume(const io::VolumeData& vol,
                                  const PointCloudParams& params) {
    PointCloud pc;
    const int X = vol.shape[0], Y = vol.shape[1], Z = vol.shape[2];
    const auto& sp = vol.spacing;

    // Collect all voxels above threshold
    for (int z = 0; z < Z; ++z)
    for (int y = 0; y < Y; ++y)
    for (int x = 0; x < X; ++x) {
        float v = vol.at(x, y, z, 0);
        if (v >= params.intensity_min && v <= params.intensity_max) {
            Point3D p;
            if (params.apply_spacing) {
                p = { x * sp[0], y * sp[1], z * sp[2] };
            } else {
                p = { (float)x, (float)y, (float)z };
            }
            pc.m_points.push_back(p);
        }
    }

    LOG_INFO("Extracted {} candidate points from volume", pc.m_points.size());

    // Subsample if too large (Vietoris-Rips is O(n²) in memory)
    if ((int)pc.m_points.size() > params.max_points) {
        std::mt19937 rng(42);
        std::shuffle(pc.m_points.begin(), pc.m_points.end(), rng);
        pc.m_points.resize(params.max_points);
        LOG_INFO("Subsampled to {} points", params.max_points);
    }

    return pc;
}

void PointCloud::exportOFF(const std::string& path) const {
    std::ofstream out(path);
    out << "OFF\n" << m_points.size() << " 0 0\n";
    for (const auto& p : m_points)
        out << p[0] << " " << p[1] << " " << p[2] << "\n";
    LOG_INFO("Point cloud exported to: {}", path);
}

} // namespace mmviz::tda
