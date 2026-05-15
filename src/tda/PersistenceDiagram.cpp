#include "tda/PersistenceDiagram.h"
#include "utils/Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mmviz::tda {

// ── Filtering ────────────────────────────────────────────────────────────────

PersistenceDiagram PersistenceDiagram::filtered(float min_persistence) const {
    PersistenceDiagram out;
    for (const auto& p : points)
        if (p.persistence() >= min_persistence)
            out.points.push_back(p);
    return out;
}

// ── Feature vector for CNN input ─────────────────────────────────────────────
// Implements a simplified persistence image: for each dimension (0,1,2),
// build a 1D histogram of persistence values → concatenate → normalize
// This is the standard way to use TDA signatures as ML features.

std::vector<float> PersistenceDiagram::toFeatureVector(int bins) const {
    // 3 dimensions × bins each
    std::vector<float> feat(3 * bins, 0.0f);

    // Find global max persistence for normalization
    float max_p = 0.0f;
    for (const auto& p : points)
        max_p = std::max(max_p, p.persistence());
    if (max_p == 0.0f) return feat;

    for (const auto& p : points) {
        int dim = std::clamp(p.dimension, 0, 2);
        float norm_p = p.persistence() / max_p;
        // Weight by persistence (more persistent features matter more)
        int bin = std::min((int)(norm_p * bins), bins - 1);
        feat[dim * bins + bin] += p.persistence();
    }

    // L2 normalize
    float norm = 0.0f;
    for (float v : feat) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0.0f)
        for (float& v : feat) v /= norm;

    return feat;
}

// ── Bottleneck distance ───────────────────────────────────────────────────────
// Exact bottleneck distance via greedy matching (approximate for large diagrams)
// For production use GUDHI::bottleneck_distance

float PersistenceDiagram::bottleneckDistance(const PersistenceDiagram& other) const {
    // Split by dimension and compute per-dimension matching cost
    float max_cost = 0.0f;

    for (int dim = 0; dim <= 2; ++dim) {
        std::vector<const PersistencePoint*> A, B;
        for (const auto& p : points)       if (p.dimension == dim) A.push_back(&p);
        for (const auto& p : other.points) if (p.dimension == dim) B.push_back(&p);

        // Greedy: for each point in A, find nearest in B (L∞ metric on diagram)
        // A proper implementation uses the Hungarian algorithm or GUDHI
        for (const auto* a : A) {
            float best = a->persistence() / 2.0f;  // cost of matching to diagonal
            for (const auto* b : B) {
                float cost = std::max(std::abs(a->birth - b->birth),
                                      std::abs(a->death - b->death));
                best = std::min(best, cost);
            }
            max_cost = std::max(max_cost, best);
        }
    }
    return max_cost;
}

// ── I/O ──────────────────────────────────────────────────────────────────────

void PersistenceDiagram::saveCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "birth,death,dimension\n";
    for (const auto& p : points)
        out << p.birth << "," << p.death << "," << p.dimension << "\n";
    LOG_INFO("Persistence diagram saved to: {}", path);
}

PersistenceDiagram PersistenceDiagram::loadCSV(const std::string& path) {
    PersistenceDiagram diag;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    std::getline(in, line); // skip header
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        PersistencePoint p;
        char comma;
        ss >> p.birth >> comma >> p.death >> comma >> p.dimension;
        diag.points.push_back(p);
    }
    return diag;
}

} // namespace mmviz::tda
