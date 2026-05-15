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

// ── Feature vector for model input (192-dim) ─────────────────────────────────
// Exact layout matches train_model.py _build_tda_vector():
//  [0:3]     H0_count, H1_count, H2_count
//  [3:7]     max_persistence, mean_persistence, std_persistence, placeholder=0
//  [7:10]    persistence_entropy H0, H1, H2
//  [10:70]   top-20 birth/death pairs for H0 (40 floats, persistence-sorted desc)
//  [70:110]  top-20 birth/death pairs for H1 (40 floats)
//  [110:150] top-20 birth/death pairs for H2 (40 floats)
//  [150:192] zeros

static float persistenceEntropy(const std::vector<std::pair<float,float>>& pairs) {
    if (pairs.empty()) return 0.0f;
    std::vector<float> pers;
    for (auto [b,d] : pairs)
        if (d > b) pers.push_back(d - b);
    if (pers.empty()) return 0.0f;
    float total = std::accumulate(pers.begin(), pers.end(), 0.0f);
    if (total < 1e-12f) return 0.0f;
    float H = 0.0f;
    for (float p : pers) {
        float r = p / total;
        H -= r * std::log(r + 1e-12f);
    }
    return H;
}

std::vector<float> PersistenceDiagram::toFeatureVector(int /*bins*/) const {
    std::vector<float> feat(192, 0.0f);

    // Split by dimension, keep only finite pairs
    std::vector<std::pair<float,float>> h0, h1, h2;
    for (const auto& p : points) {
        if (p.death <= p.birth) continue;
        auto pr = std::make_pair(p.birth, p.death);
        if      (p.dimension == 0) h0.push_back(pr);
        else if (p.dimension == 1) h1.push_back(pr);
        else if (p.dimension == 2) h2.push_back(pr);
    }

    // [0:3] counts
    feat[0] = (float)h0.size();
    feat[1] = (float)h1.size();
    feat[2] = (float)h2.size();

    // Global persistence statistics [3:7]
    std::vector<float> all_pers;
    for (auto& v : {h0, h1, h2})
        for (auto [b,d] : v) all_pers.push_back(d - b);

    if (!all_pers.empty()) {
        float sum = 0, sum2 = 0;
        for (float p : all_pers) { sum += p; sum2 += p*p; }
        float mn = sum / all_pers.size();
        feat[3] = *std::max_element(all_pers.begin(), all_pers.end());
        feat[4] = mn;
        feat[5] = std::sqrt(std::max(0.0f, sum2/all_pers.size() - mn*mn));
    }
    feat[6] = 0.0f;  // placeholder

    // [7:10] entropy per dimension
    feat[7] = persistenceEntropy(h0);
    feat[8] = persistenceEntropy(h1);
    feat[9] = persistenceEntropy(h2);

    // Top-20 birth/death pairs (desc by persistence) [10:70], [70:110], [110:150]
    auto top20flat = [&](std::vector<std::pair<float,float>>& pairs, int start) {
        std::partial_sort(pairs.begin(),
                          pairs.begin() + std::min((int)pairs.size(), 20),
                          pairs.end(),
                          [](auto& a, auto& b){ return (a.second-a.first) > (b.second-b.first); });
        int lim = std::min((int)pairs.size(), 20);
        for (int i = 0; i < lim; ++i) {
            feat[start + 2*i]   = pairs[i].first;
            feat[start + 2*i+1] = pairs[i].second;
        }
    };
    top20flat(h0,  10);
    top20flat(h1,  70);
    top20flat(h2, 110);

    // [150:192] already zero

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
