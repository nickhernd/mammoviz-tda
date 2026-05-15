#include "tda/VietorisRips.h"
#include "utils/Logger.h"
#include "utils/Timer.h"

#include <gudhi/Rips_complex.h>
#include <gudhi/Sparse_rips_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/distance_functions.h>

#include <vector>
#include <cmath>

namespace mmviz::tda {

// GUDHI 3.10 uses namespace Gudhi (capital G)
using Simplex_tree  = Gudhi::Simplex_tree<>;
using Filt          = Simplex_tree::Filtration_value;
using Dense_rips    = Gudhi::rips_complex::Rips_complex<Filt>;
using Sparse_rips   = Gudhi::rips_complex::Sparse_rips_complex<Filt>;
using Field_Zp      = Gudhi::persistent_cohomology::Field_Zp;
using Pcoh          = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;

using GPoint = std::vector<double>;

static std::vector<GPoint> toGUDHIPoints(const PointCloud& cloud) {
    std::vector<GPoint> pts;
    pts.reserve(cloud.size());
    for (const auto& p : cloud.points())
        pts.push_back({ (double)p[0], (double)p[1], (double)p[2] });
    return pts;
}

VietorisRips::VietorisRips(const RipsParams& params) : m_params(params) {}

PersistenceDiagram VietorisRips::compute(const PointCloud& cloud) const {
    utils::Timer timer;
    LOG_INFO("Building Vietoris-Rips: {} points, max_ε={:.2f}mm, max_dim={}",
             cloud.size(), m_params.max_edge_length, m_params.max_dimension);

    auto gpts = toGUDHIPoints(cloud);

    Simplex_tree simplex_tree;
    Gudhi::Euclidean_distance dist;

    if ((int)cloud.size() > m_params.sparse_threshold) {
        // Sparse Rips: (points, distance, epsilon, mini, maxi)
        // epsilon=1.0 gives a 2-approximation; maxi = max edge length
        LOG_INFO("Using sparse Rips (n={} > threshold={})", cloud.size(), m_params.sparse_threshold);
        Sparse_rips sparse(gpts, dist, 1.0, 0.0, (double)m_params.max_edge_length);
        sparse.create_complex(simplex_tree, m_params.max_dimension);
    } else {
        Dense_rips rips(gpts, (Filt)m_params.max_edge_length, dist);
        rips.create_complex(simplex_tree, m_params.max_dimension);
    }

    LOG_INFO("Simplex tree: {} simplices — computing persistence...",
             simplex_tree.num_simplices());

    Pcoh pcoh(simplex_tree, true);
    pcoh.init_coefficients(2);   // Z/2Z coefficients
    pcoh.compute_persistent_cohomology(0.0f);

    PersistenceDiagram diagram;
    for (auto& pair : pcoh.get_persistent_pairs()) {
        Filt birth = simplex_tree.filtration(std::get<0>(pair));
        Filt death = simplex_tree.filtration(std::get<1>(pair));
        int  dim   = simplex_tree.dimension(std::get<0>(pair));

        if (std::isinf((double)death) || death <= birth) continue;

        diagram.points.push_back({ (float)birth, (float)death, dim });
    }

    LOG_INFO("Persistence: {} pairs in {:.2f}s", diagram.points.size(), timer.elapsedSeconds());
    return diagram;
}

std::vector<PersistenceDiagram> VietorisRips::computeTemporal(
    const std::vector<PointCloud>& phases) const
{
    std::vector<PersistenceDiagram> result;
    result.reserve(phases.size());
    for (int i = 0; i < (int)phases.size(); ++i) {
        LOG_INFO("DCE-MRI phase {}/{}", i + 1, phases.size());
        result.push_back(compute(phases[i]));
    }
    return result;
}

} // namespace mmviz::tda
