#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <limits>
#include <cstdlib>

#include "dnf_params.hpp"   // from cpp/param_data_structure/include
#include "json.hpp"         // nlohmann::json

using nlohmann::json;

// -----------------------------------------------------------------------------
// Config: paths + python command (same idea as NSGA-II file)
// -----------------------------------------------------------------------------

static const std::string CONFIG_PATH       = "candidate_dnf_config.yaml";
static const std::string METRICS_PATH      = "candidate_metrics.json";
static const std::string PYTHON_CMD_PREFIX = "python toy_eval.py ";

// -----------------------------------------------------------------------------
// Metrics + individual representation
// -----------------------------------------------------------------------------

struct Metrics {
    double k90_ms   = std::numeric_limits<double>::quiet_NaN();
    double k99_ms   = std::numeric_limits<double>::quiet_NaN();
    double ripple   = std::numeric_limits<double>::quiet_NaN();
    double peak     = std::numeric_limits<double>::quiet_NaN();
    double steps    = std::numeric_limits<double>::quiet_NaN();
    bool   pass     = false;
    bool   valid    = false;
};

struct Individual {
    DNFParams params;
    Metrics   metrics;

    std::vector<double> objectives;  // [k90, ripple, steps, overshoot]
    double scalar_obj = std::numeric_limits<double>::infinity(); // sum(objectives)
};

static std::mt19937& globalRNG() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// -----------------------------------------------------------------------------
// Helpers: write config, run python, read metrics
// -----------------------------------------------------------------------------

void write_config_yaml(const DNFParams& p, const std::string& path) {
    json j = p.toJSON();
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[ERROR] Could not open " << path << " for writing config.\n";
        return;
    }
    ofs << j.dump(2);
}

bool read_metrics_json(const std::string& path, Metrics& m) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "[ERROR] Could not open metrics file: " << path << "\n";
        m.valid = false;
        return false;
    }

    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse metrics JSON: " << e.what() << "\n";
        m.valid = false;
        return false;
    }

    m.k90_ms = j.value("k90_ms", 1e9);
    m.k99_ms = j.value("k99_ms", 1e9);
    m.ripple = j.value("ripple", 1e9);
    m.peak   = j.value("peak",   1e9);
    m.steps  = j.value("steps",  1e9);
    m.pass   = j.value("pass",   false);
    m.valid  = true;
    return true;
}

Metrics run_python_eval(const DNFParams& p) {
    Metrics m;

    // 1) write config
    write_config_yaml(p, CONFIG_PATH);

    // 2) call python
    std::string cmd = PYTHON_CMD_PREFIX + CONFIG_PATH + " " + METRICS_PATH;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "[WARN] Python eval returned non-zero code " << rc << "\n";
        m.valid = false;
        return m;
    }

    // 3) read metrics
    if (!read_metrics_json(METRICS_PATH, m)) {
        m.valid = false;
    }

    return m;
}

// -----------------------------------------------------------------------------
// Objectives mapping (same as NSGA-II file)
// -----------------------------------------------------------------------------

void fill_objectives_from_metrics(const Metrics& m, Individual& ind) {
    ind.objectives.clear();
    const double BIG_PENALTY = 1e6;

    double obj_k90        = m.k90_ms;
    double obj_ripple     = m.ripple;
    double obj_steps      = m.steps;
    double obj_overshoot  = std::max(0.0, m.peak - 1.0);

    ind.objectives.push_back(obj_k90);
    ind.objectives.push_back(obj_ripple);
    ind.objectives.push_back(obj_steps);
    ind.objectives.push_back(obj_overshoot);

    if (!m.valid || !m.pass) {
        for (double& v : ind.objectives) {
            v += BIG_PENALTY;
        }
    }

    ind.scalar_obj = std::accumulate(ind.objectives.begin(),
                                     ind.objectives.end(), 0.0);
}

// Evaluate one individual completely
void evaluate_individual(Individual& ind) {
    ind.metrics = run_python_eval(ind.params);
    fill_objectives_from_metrics(ind.metrics, ind);
}

// -----------------------------------------------------------------------------
// "MOTPE-style" sampler: sample around good points
// -----------------------------------------------------------------------------

double sample_gaussian_clamped(double mean,
                               double lo,
                               double hi,
                               double sigma_frac = 0.15) {
    std::mt19937& gen = globalRNG();
    double range = hi - lo;
    double sigma = sigma_frac * range;
    if (sigma <= 0.0) return mean;

    std::normal_distribution<double> dist(mean, sigma);
    for (int tries = 0; tries < 20; ++tries) {
        double v = dist(gen);
        if (v >= lo && v <= hi) return v;
    }
    // fallback: clamp mean
    return std::min(std::max(mean, lo), hi);
}

DNFParams propose_from_good(const std::vector<Individual>& archive,
                            const std::vector<int>& good_indices) {
    std::mt19937& gen = globalRNG();
    std::uniform_int_distribution<int> pick_idx(0,
        static_cast<int>(good_indices.size()) - 1);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    // pick a "good" base point
    const DNFParams& base = archive[good_indices[pick_idx(gen)]].params;

    DNFParams c;

    // continuous fields: sample Gaussian around base, clamped to allowed ranges
    c.tau   = sample_gaussian_clamped(base.tau,   0.005, 0.020);
    c.dt    = sample_gaussian_clamped(base.dt,    0.0005, 0.0020);
    c.beta  = sample_gaussian_clamped(base.beta,  2.0,   7.0);
    c.theta = sample_gaussian_clamped(base.theta, 0.0,   0.1);
    c.h     = sample_gaussian_clamped(base.h,    -0.60, -0.05);
    c.w_exc = sample_gaussian_clamped(base.w_exc, 0.8,   3.0);
    c.sigma_exc = sample_gaussian_clamped(base.sigma_exc, 2.0, 6.0);
    c.w_inh     = sample_gaussian_clamped(base.w_inh,     0.3, 2.0);

    // sigma_inh must be > sigma_exc + 1 and [3, 20]
    {
        double lo = std::max(3.0, c.sigma_exc + 1.0000001);
        double hi = 20.0;
        if (lo >= hi) lo = hi - 1e-3; // just in case
        c.sigma_inh = sample_gaussian_clamped(base.sigma_inh, lo, hi);
    }

    c.intensity_scale_factor =
        sample_gaussian_clamped(base.intensity_scale_factor, 3.0, 12.0);

    // discrete fields: copy base most of the time, or random jump
    if (u01(gen) < 0.8) {
        c.gaussian_stimulus_width = base.gaussian_stimulus_width;
    } else {
        std::uniform_int_distribution<int> gw(3, 11);
        c.gaussian_stimulus_width = gw(gen);
    }

    if (u01(gen) < 0.8) {
        c.N = base.N;
    } else {
        const int options[4] = {50, 100, 150, 200};
        std::uniform_int_distribution<int> idx(0, 3);
        c.N = options[idx(gen)];
    }

    // if invalid (alpha out of range etc.), just fall back to randomSample
    if (!c.isValid()) {
        c = DNFParams::randomSample();
    }
    return c;
}

DNFParams propose_candidate(const std::vector<Individual>& archive) {
    // If we don’t have enough history yet, just randomSample.
    if (archive.size() < 10) {
        return DNFParams::randomSample();
    }

    // Sort indices by scalar objective (ascending)
    std::vector<int> idx(archive.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  return archive[a].scalar_obj < archive[b].scalar_obj;
              });

    // "Good" set = top q fraction (e.g., 25%) with a minimum size
    const double q = 0.25;
    int n_good = static_cast<int>(std::max(
        5.0,
        std::floor(q * static_cast<double>(archive.size()))
    ));
    n_good = std::min(n_good, static_cast<int>(archive.size()));

    std::vector<int> good_indices(idx.begin(), idx.begin() + n_good);

    return propose_from_good(archive, good_indices);
}

// -----------------------------------------------------------------------------
// MAIN: MOTPE-style adaptive search
// -----------------------------------------------------------------------------

int main() {
    std::cout << "=== MOTPE-style multi-objective DNF search (C++) ===\n\n";

    const int N_EVAL = 200;  // total evaluations / trials

    std::vector<Individual> archive;
    archive.reserve(N_EVAL);

    for (int t = 0; t < N_EVAL; ++t) {
        Individual ind;

        // sample params
        if (archive.empty()) {
            ind.params = DNFParams::randomSample();
        } else {
            ind.params = propose_candidate(archive);
        }

        // evaluate via Python toy_eval
        evaluate_individual(ind);
        archive.push_back(ind);

        std::cout << "Trial " << t
                  << " | scalar_obj=" << ind.scalar_obj
                  << " | pass=" << (ind.metrics.pass ? "true" : "false")
                  << "\n";
    }

    // Pick best by scalar objective
    if (archive.empty()) {
        std::cerr << "[ERROR] archive is empty?!\n";
        return 1;
    }

    auto best_it = std::min_element(
        archive.begin(), archive.end(),
        [](const Individual& a, const Individual& b) {
            return a.scalar_obj < b.scalar_obj;
        });

    const Individual& best = *best_it;

    std::cout << "\n=== Search finished (MOTPE-style) ===\n";
    std::cout << "Best scalarized objective sum: " << best.scalar_obj << "\n";
    std::cout << "Best metrics:\n";
    std::cout << "  k90_ms = " << best.metrics.k90_ms << "\n";
    std::cout << "  k99_ms = " << best.metrics.k99_ms << "\n";
    std::cout << "  ripple = " << best.metrics.ripple << "\n";
    std::cout << "  peak   = " << best.metrics.peak << "\n";
    std::cout << "  steps  = " << best.metrics.steps << "\n";
    std::cout << "  pass   = " << (best.metrics.pass ? "true" : "false") << "\n\n";

    std::cout << "Best params (JSON):\n";
    std::cout << best.params.toJSON().dump(2) << "\n";
    std::cout << "=== End of MOTPE-style DNF search ===\n";

    return 0;
}
