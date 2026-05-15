#pragma once
#include <vector>
#include <array>
#include <cstdint>

namespace mmviz::io {

// Stores a 3D or 4D medical volume
// For DCE-MRI: shape = {X, Y, Z, T}
// For DBT mammography: shape = {X, Y, Z, 1}
struct VolumeData {
    std::array<int, 4>  shape;      // {X, Y, Z, T}
    std::array<float,3> spacing;    // voxel spacing in mm
    std::vector<float>  data;       // row-major flat buffer

    float at(int x, int y, int z, int t = 0) const;
    int   voxelCount() const;
    float* rawPtr() { return data.data(); }

    // Returns temporal contrast curve at voxel (x,y,z) — relevant for DCE-MRI
    std::vector<float> temporalCurve(int x, int y, int z) const;
};

} // namespace mmviz::io
