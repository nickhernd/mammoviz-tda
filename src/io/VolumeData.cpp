#include "io/VolumeData.h"
#include <stdexcept>
#include <numeric>

namespace mmviz::io {

float VolumeData::at(int x, int y, int z, int t) const {
    int idx = t * shape[0]*shape[1]*shape[2]
            + z * shape[0]*shape[1]
            + y * shape[0]
            + x;
    return data[idx];
}

int VolumeData::voxelCount() const {
    return shape[0] * shape[1] * shape[2] * shape[3];
}

std::vector<float> VolumeData::temporalCurve(int x, int y, int z) const {
    std::vector<float> curve(shape[3]);
    for (int t = 0; t < shape[3]; ++t)
        curve[t] = at(x, y, z, t);
    return curve;
}

} // namespace mmviz::io
