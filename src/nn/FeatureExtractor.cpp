#include "nn/FeatureExtractor.h"
#include "utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

namespace mmviz::nn {

FeatureExtractor::FeatureExtractor(const FeatureExtractorParams& p) : m_params(p) {}

static float percentile_sorted(const std::vector<float>& sv, float pct) {
    if (sv.empty()) return 0.0f;
    float idx = pct * 0.01f * (float)(sv.size() - 1);
    int lo = (int)idx;
    int hi = std::min(lo + 1, (int)sv.size() - 1);
    float t = idx - lo;
    return sv[lo] * (1.0f - t) + sv[hi] * t;
}

// 6-connected BFS component count on 3D boolean mask
static int countComponents(const std::vector<bool>& mask, int X, int Y, int Z) {
    std::vector<bool> visited(mask.size(), false);
    const int dx[] = {1,-1,0,0,0,0};
    const int dy[] = {0,0,1,-1,0,0};
    const int dz[] = {0,0,0,0,1,-1};
    int count = 0;
    for (int i = 0; i < X*Y*Z; ++i) {
        if (!mask[i] || visited[i]) continue;
        ++count;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            int cx =  cur % X;
            int cy = (cur / X) % Y;
            int cz =  cur / (X * Y);
            for (int d = 0; d < 6; ++d) {
                int nx = cx+dx[d], ny = cy+dy[d], nz = cz+dz[d];
                if (nx<0||nx>=X||ny<0||ny>=Y||nz<0||nz>=Z) continue;
                int ni = nz*X*Y + ny*X + nx;
                if (mask[ni] && !visited[ni]) { visited[ni]=true; q.push(ni); }
            }
        }
    }
    return count;
}

// ── Exact match of train_model.py extract_image_features() ───────────────────
// Layout (512-dim):
//  [0:64]   64-bin intensity histogram, normalized by voxel count
//  [64:77]  13 stats: percentiles (1,5,10,25,50,75,90,95,99) + mean,std,skew,kurt
//  [77:86]  9 calcification stats (intensity>0.85)
//  [86:90]  4 texture stats: gradient mean/std/p90, mass-region variance
//  [90:512] zeros
std::vector<float> FeatureExtractor::extract(const io::VolumeData& vol) const {
    const int X = vol.shape[0], Y = vol.shape[1], Z = vol.shape[2];
    const int N = X * Y * Z;

    // Flat copy for statistics
    std::vector<float> flat(vol.data.begin(), vol.data.begin() + N);
    std::vector<float> sv = flat;
    std::sort(sv.begin(), sv.end());

    std::vector<float> feats(512, 0.0f);

    // ── [0:64] normalized histogram ───────────────────────────────────────────
    for (float v : flat) {
        int bin = std::clamp((int)(v * 64.0f), 0, 63);
        feats[bin] += 1.0f;
    }
    if (N > 0)
        for (int i = 0; i < 64; ++i) feats[i] /= (float)N;

    // ── [64:77] percentile + moment stats ─────────────────────────────────────
    const float pcts[] = {1,5,10,25,50,75,90,95,99};
    for (int i = 0; i < 9; ++i)
        feats[64+i] = percentile_sorted(sv, pcts[i]);

    double mean = 0.0;
    for (float v : flat) mean += v;
    mean /= N;

    double var = 0.0;
    for (float v : flat) { double d = v - mean; var += d*d; }
    var /= N;
    double stdv = std::sqrt(var + 1e-12);

    double skw = 0.0, kurt = 0.0;
    for (float v : flat) {
        double d = (v - mean) / stdv;
        double d2 = d*d;
        skw  += d2*d;
        kurt += d2*d2;
    }
    skw  /= N;
    kurt  = kurt/N - 3.0;

    feats[73] = (float)mean;
    feats[74] = (float)stdv;
    feats[75] = (float)skw;
    feats[76] = (float)kurt;

    // ── [77:86] calcification stats (intensity > 0.85) ────────────────────────
    std::vector<bool> calc_mask(N, false);
    std::vector<float> cx, cy, cz;

    for (int z = 0; z < Z; ++z)
    for (int y = 0; y < Y; ++y)
    for (int x = 0; x < X; ++x) {
        if (vol.at(x,y,z) > 0.85f) {
            int idx = z*X*Y + y*X + x;
            calc_mask[idx] = true;
            cx.push_back((float)x);
            cy.push_back((float)y);
            cz.push_back((float)z);
        }
    }

    float count = (float)cx.size();
    feats[77] = count;

    if (count >= 3.0f) {
        int cn = (int)cx.size();

        auto vmean = [](const std::vector<float>& v) {
            float s = 0; for (float x : v) s += x; return s / v.size();
        };
        float mx = vmean(cx), my = vmean(cy), mz = vmean(cz);
        feats[78] = mx; feats[79] = my; feats[80] = mz;

        auto vstd = [](const std::vector<float>& v, float m) {
            float s = 0; for (float x : v) { float d=x-m; s+=d*d; }
            return std::sqrt(s/v.size() + 1e-8f);
        };
        feats[81] = vstd(cx, mx);
        feats[82] = vstd(cy, my);
        feats[83] = vstd(cz, mz);

        // Linearity: ratio largest/smallest diagonal of covariance (approx eigenvalues)
        double Cxx=0, Cyy=0, Czz=0;
        for (int i = 0; i < cn; ++i) {
            double dx=cx[i]-mx, dy=cy[i]-my, dz=cz[i]-mz;
            Cxx+=dx*dx; Cyy+=dy*dy; Czz+=dz*dz;
        }
        Cxx/=cn; Cyy/=cn; Czz/=cn;
        double e_max = std::max({Cxx, Cyy, Czz}) + 1e-8;
        double e_min = std::min({Cxx, Cyy, Czz}) + 1e-8;
        feats[84] = (float)(e_max / e_min);

        feats[85] = (float)countComponents(calc_mask, X, Y, Z);
    }

    // ── [86:90] texture stats ─────────────────────────────────────────────────
    std::vector<float> grad(N, 0.0f);
    for (int z = 1; z < Z-1; ++z)
    for (int y = 1; y < Y-1; ++y)
    for (int x = 1; x < X-1; ++x) {
        float gx = vol.at(x+1,y,z) - vol.at(x-1,y,z);
        float gy = vol.at(x,y+1,z) - vol.at(x,y-1,z);
        float gz = vol.at(x,y,z+1) - vol.at(x,y,z-1);
        grad[z*X*Y + y*X + x] = std::sqrt(gx*gx + gy*gy + gz*gz);
    }

    double gmean = 0, gvar = 0;
    for (float v : grad) gmean += v;
    gmean /= N;
    for (float v : grad) { double d=v-gmean; gvar+=d*d; }
    gvar /= N;

    std::vector<float> sg = grad;
    std::sort(sg.begin(), sg.end());
    float gp90 = percentile_sorted(sg, 90.0f);

    double mass_s = 0, mass_sq = 0;
    int mass_n = 0;
    for (float v : flat)
        if (v > 0.3f && v < 0.8f) { mass_s += v; mass_sq += v*v; ++mass_n; }
    float mass_var = 0.0f;
    if (mass_n > 0) {
        double m = mass_s / mass_n;
        mass_var = (float)(mass_sq/mass_n - m*m);
    }

    feats[86] = (float)gmean;
    feats[87] = (float)std::sqrt(gvar);
    feats[88] = gp90;
    feats[89] = mass_var;

    // [90:512] already zero

    LOG_INFO("Extracted 512 image features (histogram+stats+calcs+texture)");
    return feats;
}

} // namespace mmviz::nn
