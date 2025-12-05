#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <limits>
#include <cstdlib>

#include "dnf_params.hpp"   // Your DNFParams struct + randomSample + isValid
#include "json.hpp"         // nlohmann::json, sitting in the nlohmann include dir

using nlohmann::json;

// -----------------------------------------------------------------------------
// Config: paths + python command
// -----------------------------------------------------------------------------

// We will reuse the same evaluation style as the NSGA search.
// If your toy_eval.py lives in playground/toy_search_pipeline, use this:
static const std::string CONFIG_PATH  = "candidate_dnf_config.yaml";
static const std::string METRICS_PATH = "candidate_metrics.json";
static const std::string PYTHON_CMD_PREFIX =
    "python playground/toy_search_pipeline/toy_eval.py ";

// If later you want to switch to the real DNF wrapper, you only change this:
// static const std::string PYTHON_CMD_PREFIX =
//     "python python_env/dnf_eval_wrapper.py ";

// -----------------------------------------------------------------------------
// Metrics + trials
// -----------------------------------------------------------------------------

struct Metrics {
    double k90_ms   = std::numeric_limits<double>::quiet_NaN();
    double k99_ms   = std::numeric_limits<double>::quiet_NaN();
    double ripple   = std::numeric_limits<double>::quiet_NaN();
    double peak     = std::numeric_limits<double>::quiet_NaN();
    double steps    = std::numeric_limits<double>::quiet_NaN();
    bool   pass     = false;
    bool   valid    = false;   // did we successfully read + parse metrics?
};

struct Trial {
    DNFParams params;
    Metrics   metrics;
    double    scalar_obj = std::numeric_limits<double>::infinity();
};

// global RNG
static std::mt19937& globalRNG() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// -----------------------------------------------------------------------------
// Helpers: write config, run python, read metrics
// -----------------------------------------------------------------------------

void write_config_yaml(const DNFParams& p, const std::string& path) {
    json j = p.toJSON();  // already implemented in dnf_params.cpp
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[ERROR] Could not open " << path << " for writing config.\n";
        return;
    }
    // JSON syntax is valid YAML; toy_eval.py can json.load or yaml.safe_load.
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
// Scalar objective for MOTPE-style search
// -----------------------------------------------------------------------------

double scalar_objective(const Metrics& m) {
    // Big penalty if invalid or gate failed
    if (!m.valid || !m.pass) {
        return 1e9;
    }

    // We want:
    //  - k90_ms small
    //  - ripple small
    //  - steps small
    //  - avoid peak overshoot far above 1
    double overshoot = std::max(0.0, m.peak - 1.0);

    // Simple weighted sum (tune weights if you like):
    double obj =
        m.k90_ms +
        100.0 * m.ripple +
        0.05 * m.steps +
        200.0 * overshoot;

    return obj;
}

// -----------------------------------------------------------------------------
// MOTPE-style sampling (very simplified, but biased toward good trials)
// -----------------------------------------------------------------------------

// Helper: clamp a value
static double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// Approximate "ranges" for the parameters (copied from isValid logic)
struct ParamRange {
    double lo;
    double hi;
};

static ParamRange tau_range   {0.005, 0.020};
static ParamRange dt_range    {0.0005, 0.0020};
static ParamRange beta_range  {2.0, 7.0};
static ParamRange theta_range {0.0, 0.1};
static ParamRange h_range     {-0.60, -0.05};
static ParamRange w_exc_range {0.8, 3.0};
static ParamRange sigma_exc_range{2.0, 6.0};
static ParamRange w_inh_range {0.3, 2.0};
// sigma_inh also has constraint sigma_inh > sigma_exc + 1, we'll handle later
static ParamRange sigma_inh_range{3.0, 20.0};
static ParamRange intensity_range{3.0, 12.0};

// Allowed discrete options
static const int N_options[] = {50, 100, 150, 200};

// Add some noise around a base value (normal), clamped to range.
double perturb(double base, ParamRange r, double frac = 0.15) {
    std::mt19937& gen = globalRNG();
    double range = r.hi - r.lo;
    double sigma = frac * range;
    std::normal_distribution<double> dist(base, sigma);
    for (int tries = 0; tries < 20; ++tries) {
        double v = dist(gen);
        if (v >= r.lo && v <= r.hi) return v;
    }
    // fallback
    return clamp(base, r.lo, r.hi);
}

// sample an integer in [lo, hi]
int rand_int(int lo, int hi) {
    std::mt19937& gen = globalRNG();
    std::uniform_int_distribution<int> d(lo, hi);
    return d(gen);
}

// Choose a "good" base trial from top q% best, then perturb around it.
DNFParams sample_motpe_candidate(const std::vector<Trial>& trials) {
    const size_t n = trials.size();
    if (n == 0) {
        return DNFParams::randomSample();
    }

    // Sort indices by objective ascending
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  return trials[a].scalar_obj < trials[b].scalar_obj;
              });

    // "good" set = best q% or at least some minimum
    const double q = 0.2;
    size_t good_size = std::max<size_t>(3, static_cast<size_t>(q * n));
    if (good_size > n) good_size = n;

    std::mt19937& gen = globalRNG();
    std::uniform_int_distribution<int> pick_good(0,
        static_cast<int>(good_size) - 1);

    const Trial& base_trial = trials[idx[pick_good(gen)]];
    DNFParams base = base_trial.params;
    DNFParams c = base;

    // Perturb each continuous param a bit around base
    c.tau    = perturb(base.tau,   tau_range);
    c.dt     = perturb(base.dt,    dt_range);
    c.beta   = perturb(base.beta,  beta_range);
    c.theta  = perturb(base.theta, theta_range);
    c.h      = perturb(base.h,     h_range);
    c.w_exc  = perturb(base.w_exc, w_exc_range);
    c.sigma_exc = perturb(base.sigma_exc, sigma_exc_range);
    c.w_inh  = perturb(base.w_inh, w_inh_range);

    // sigma_inh must be > sigma_exc + 1 and within [3,20]
    double sigma_lo = std::max(sigma_inh_range.lo, c.sigma_exc + 1.0000001);
    double sigma_hi = sigma_inh_range.hi;
    if (sigma_lo > sigma_hi) {
        // fallback: resample completely later
        c = DNFParams::randomSample();
    } else {
        ParamRange sigma_inh_local{sigma_lo, sigma_hi};
        c.sigma_inh = perturb(base.sigma_inh, sigma_inh_local);
    }

    c.intensity_scale_factor = perturb(base.intensity_scale_factor,
                                       intensity_range);

    // Discrete params: pick from the base or "good" neighbours
    c.gaussian_stimulus_width = base.gaussian_stimulus_width;
    if (rand_int(0, 3) == 0) { // 1/4 chance tweak width
        int delta = rand_int(-1, 1);
        int w = c.gaussian_stimulus_width + delta;
        if (w < 3) w = 3;
        if (w > 11) w = 11;
        c.gaussian_stimulus_width = w;
    }

    // N: choose one of allowed values, biased toward base
    int idxN = 0;
    for (int i = 0; i < 4; ++i) {
        if (N_options[i] == base.N) { idxN = i; break; }
    }
    if (rand_int(0, 3) == 0) { // tweak sometimes
        int delta = rand_int(-1, 1);
        idxN = clamp(idxN + delta, 0, 3);
    }
    c.N = N_options[idxN];

    // If somehow invalid, fall back to a fresh random sample
    if (!c.isValid()) {
        c = DNFParams::randomSample();
    }
    return c;
}

// -----------------------------------------------------------------------------
// MAIN (MOTPE-style search)
// -----------------------------------------------------------------------------

int main() {
    std::cout << "=== MOTPE-style DNF search ===\n\n";

    const int N_TRIALS      = 200;  // total evaluations
    const int N_RANDOM_INIT = 20;   // purely random warm-up

    std::vector<Trial> trials;
    trials.reserve(N_TRIALS);

    double best_obj = std::numeric_limits<double>::infinity();
    int best_idx = -1;

    for (int t = 0; t < N_TRIALS; ++t) {
        Trial trial;

        if (t < N_RANDOM_INIT || trials.empty()) {
            trial.params = DNFParams::randomSample();
        } else {
            trial.params = sample_motpe_candidate(trials);
        }

        trial.metrics = run_python_eval(trial.params);
        trial.scalar_obj = scalar_objective(trial.metrics);

        trials.push_back(trial);

        if (trial.scalar_obj < best_obj) {
            best_obj = trial.scalar_obj;
            best_idx = static_cast<int>(trials.size()) - 1;
        }

        std::cout << "Trial " << t
                  << " | scalar_obj=" << trial.scalar_obj
                  << " | pass=" << (trial.metrics.pass ? "true" : "false")
                  << "\n";
    }

    std::cout << "\n=== Search finished (MOTPE-style) ===\n";
    std::cout << "Best scalarized objective sum: " << best_obj << "\n";
    std::cout << "Best metrics:\n";

    if (best_idx >= 0) {
        const Trial& best = trials[best_idx];
        std::cout << "  k90_ms = " << best.metrics.k90_ms << "\n";
        std::cout << "  k99_ms = " << best.metrics.k99_ms << "\n";
        std::cout << "  ripple = " << best.metrics.ripple << "\n";
        std::cout << "  peak   = " << best.metrics.peak << "\n";
        std::cout << "  steps  = " << best.metrics.steps << "\n";
        std::cout << "  pass   = " << (best.metrics.pass ? "true" : "false") << "\n\n";

        std::cout << "Best params (JSON):\n";
        std::cout << best.params.toJSON().dump(2) << "\n";
    } else {
        std::cout << "  (no valid trials)\n";
    }

    std::cout << "=== End of MOTPE-style DNF search ===\n";
    return 0;
}
