#pragma once
#include <vector>
#include <array>
#include <string>

namespace mmviz::tda {

// A single topological feature: (birth_ε, death_ε, dimension)
// dimension: 0 = connected components (β₀)
//            1 = loops/tunnels          (β₁)
//            2 = voids/cavities         (β₂)
struct PersistencePoint {
    float birth;
    float death;
    int   dimension;

    float persistence() const { return death - birth; }
    // Points with high persistence are topologically significant (not noise)
};

// Full persistence diagram for one filtration
struct PersistenceDiagram {
    std::vector<PersistencePoint> points;

    // Filter by minimum persistence (noise removal)
    PersistenceDiagram filtered(float min_persistence) const;

    // Statistical descriptors usable as CNN input features
    std::vector<float> toFeatureVector(int bins = 64) const;

    // Bottleneck distance to another diagram (stability theorem)
    float bottleneckDistance(const PersistenceDiagram& other) const;

    void saveCSV(const std::string& path) const;
    static PersistenceDiagram loadCSV(const std::string& path);
};

} // namespace mmviz::tda
