#include "tda/VietorisRips.h"
#include "utils/Logger.h"
#include "utils/Timer.h"

// GUDHI headers
#include <gudhi/Rips_complex.h>
#include <gudhi/Sparse_rips_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/distance_functions.h>

#include <vector>
#include <array>

namespace mmviz::tda {

// GUDHI type aliases
using Simplex_tree         = gudhi::Simplex_tree<>;
using Filtration_value     = Simplex_tree::Filtration_value;
using Dense_rips           = gudhi::Rips_complex<Filtration_value>;
using Sparse_rips          = gudhi::Sparse_rips_complex<Filtration_value>;
using Field_Zp             = gudhi::persistent_cohomology::Field_Zp;
using Persistent_cohomology = gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;

// Convert our PointCloud to GUDHI's expected point format
using GPoint = std::vector<double>;

static std::vector<GPoint> toGUDHIPoints(const PointCloud& cloud) {
    std::vector<GPoint> pts;
    pts.reserve(cloud.size());
    for (const auto& p : cloud.points())
        pts.push_back({ (double)p[0], (double)p[1], (double)p[2] });
    return pts;
}

VietorisRips::VietorisRips(const Params& params) : m_params(params) {}

PersistenceDiagram VietorisRips::compute(const PointCloud& cloud) const {
    utils::Timer timer;
    LOG_INFO("Building Vietoris-Rips complex: {} points, max_ε={:.2f}mm, max_dim={}",
             cloud.size(), m_params.max_edge_length, m_params.max_dimension);

    auto gpts = toGUDHIPoints(cloud);

    Simplex_tree simplex_tree;

    if ((int)cloud.size() > m_params.sparse_threshold) {
        // Sparse Rips: faster for large point clouds, trades exactness for speed
        // epsilon=1.0 gives a 2-approximation of the true Rips complex
        LOG_INFO("Using sparse Rips (n={} > threshold={})", cloud.size(), m_params.sparse_threshold);
        Sparse_rips sparse(gpts, 1.0, m_params.max_edge_length);
        sparse.create_complex(simplex_tree, m_params.max_dimension);
    } else {
        Dense_rips rips(gpts, m_params.max_edge_length, gudhi::Euclidean_distance());
        rips.create_complex(simplex_tree, m_params.max_dimension);
    }

    LOG_INFO("Simplex tree: {} simplices — computing persistent cohomology...",
             simplex_tree.num_simplices());

    Persistent_cohomology pcoh(simplex_tree, true);
    pcoh.init_coefficients(2);  // Z/2Z coefficients (standard for topology)
    pcoh.compute_persistent_cohomology(0.0f);

    // Extract birth/death pairs
    PersistenceDiagram diagram;
    for (auto& pair : pcoh.get_persistent_pairs()) {
        float birth = simplex_tree.filtration(get<0>(pair));
        float death = simplex_tree.filtration(get<1>(pair));
        int   dim   = simplex_tree.dimension(get<0>(pair));

        // Skip infinite bars (features that never die within the filtration)
        // and zero-persistence pairs (numerical noise)
        if (std::isinf(death) || death <= birth) continue;

        diagram.points.push_back({ birth, death, dim });
    }

    LOG_INFO("Computed {} persistence pairs in {:.2f}s",
             diagram.points.size(), timer.elapsedSeconds());

    return diagram;
}

std::vector<PersistenceDiagram> VietorisRips::computeTemporal(
    const std::vector<PointCloud>& phases) const
{
    std::vector<PersistenceDiagram> result;
    result.reserve(phases.size());

    for (int i = 0; i < (int)phases.size(); ++i) {
        LOG_INFO("Processing DCE-MRI phase {}/{}", i + 1, phases.size());
        result.push_back(compute(phases[i]));
    }

    return result;
}

} // namespace mmviz::tda
