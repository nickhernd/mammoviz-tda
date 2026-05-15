#pragma once
#include "tda/PointCloud.h"
#include "tda/PersistenceDiagram.h"

namespace mmviz::tda {

struct RipsParams {
    float max_edge_length  = 5.0f;
    int   max_dimension    = 2;
    int   sparse_threshold = 5000;
};

// Computes persistent homology via Vietoris-Rips filtration using GUDHI
class VietorisRips {
public:
    explicit VietorisRips(const RipsParams& params = RipsParams{});

    PersistenceDiagram compute(const PointCloud& cloud) const;

    std::vector<PersistenceDiagram> computeTemporal(
        const std::vector<PointCloud>& phases) const;

private:
    RipsParams m_params;
};

} // namespace mmviz::tda
