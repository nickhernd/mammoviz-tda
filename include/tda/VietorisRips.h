#pragma once
#include "tda/PointCloud.h"
#include "tda/PersistenceDiagram.h"

namespace mmviz::tda {

// Computes persistent homology via Vietoris-Rips filtration using GUDHI
// For each ε, builds the Rips complex: edge (p,q) ∈ K_ε iff d(p,q) ≤ ε
// Tracks birth/death of topological features across the filtration
class VietorisRips {
public:
    struct Params {
        float  max_edge_length  = 5.0f;   // max ε in mm (data-dependent)
        int    max_dimension    = 2;       // compute H₀, H₁, H₂
        int    sparse_threshold = 5000;    // use sparse Rips above this size
    };

    explicit VietorisRips(const Params& params = {});

    // Main computation: P → persistence diagram
    // Uses GUDHI::rips_complex internally
    PersistenceDiagram compute(const PointCloud& cloud) const;

    // For DCE-MRI: compute diagram per time phase and track topological changes
    std::vector<PersistenceDiagram> computeTemporal(
        const std::vector<PointCloud>& phases) const;

private:
    Params m_params;
};

} // namespace mmviz::tda
