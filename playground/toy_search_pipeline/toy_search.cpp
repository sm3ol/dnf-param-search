#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <limits>

#include "dnf_params.hpp"
#include "json.hpp"

using nlohmann::json;

// ------------------------------------------------------------
// Metric struct – must match what toy_eval.py prints
// ------------------------------------------------------------
struct Metrics {
    double k90_ms   = 0.0;
    double k99_ms   = 0.0;
    double ripple   = 0.0;
    double peak     = 0.0;
    double steps    = 0.0;
    bool   pass_gate = false;  // overall certification gate
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static std::mt19937& globalRng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

double randUniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(globalRng());
}

// ------------------------------------------------------------
// Call Python: write params.json, run toy_eval.py, read metrics
// ------------------------------------------------------------
bool evalCandidate(const DNFParams& p, Metrics& out) {
    // 1) dump params to JSON file
    json j = p.toJSON();
    {
        std::ofstream ofs("params.json");
        if (!ofs) {
            std::cerr << "[C++] ERROR: cannot open params.json for writing.\n";
            return false;
        }
        ofs << j.dump(2);
    }

    // 2) call python; redirect stdout to metrics.json
    //    (toy_eval.py prints a JSON dict to stdout)
#ifdef _WIN32
    int ret = std::system("python toy_eval.py params.json > metrics.json");
#else
    int ret = std::system("python3 toy_eval.py params.json > metrics.json");
#endif
    if (ret != 0) {
        std::cerr << "[C++] ERROR: python toy_eval.py returned non-zero exit code: "
                  << ret << "\n";
        return false;
    }

    // 3) read metrics.json
    std::ifstream ifs("metrics.json");
    if (!ifs) {
        std::cerr << "[C++] ERROR: cannot open metrics.json for reading.\n";
        return false;
    }

    json m;
    try {
        ifs >> m;
    } catch (const std::exception& e) {
        std::cerr << "[C++] ERROR: failed to parse metrics.json: " << e.what() << "\n";
        return false;
    }

    try {
        out.k90_ms    = m.at("k90_ms").get<double>();
        out.k99_ms    = m.at("k99_ms").get<double>();
        out.ripple    = m.at("ripple").get<double>();
        out.peak      = m.at("peak").get<double>();
        out.steps     = m.at("steps").get<double>();
        out.pass_gate = m.at("pass").get<bool>();
    } catch (const std::exception& e) {
        std::cerr << "[C++] ERROR: missing/invalid fields in metrics.json: " << e.what() << "\n";
        return false;
    }

    return true;
}

// ------------------------------------------------------------
// Scalar fitness from multi-metric spec
//  - minimize k90_ms, ripple, steps
//  - enforce gates: k99 <= 100, peak >= 0.60, ripple <= 2
// ------------------------------------------------------------
double computeFitness(const Metrics& m) {
    // If gates fail, slap on a big penalty
    if (!m.pass_gate) {
        double penalty = 0.0;

        if (m.k99_ms > 100.0) {
            double d = m.k99_ms - 100.0;
            penalty += d * d;                     // ms^2
        }
        if (m.peak < 0.60) {
            double d = 0.60 - m.peak;
            penalty += 1000.0 * d * d;           // strong penalty if peak too low
        }
        if (m.ripple > 2.0) {
            double d = m.ripple - 2.0;
            penalty += 500.0 * d * d;            // penalize unstable tail
        }

        return 1e5 + penalty;                    // huge so gated-out points lose
    }

    // If gates pass, use a weighted sum of normalized objectives
    // Rough normalizations (you can tune later):
    double k90_norm   = m.k90_ms / 100.0;        // want < 1
    double ripple_norm= m.ripple / 2.0;          // want < 1
    double steps_norm = m.steps / 1000.0;        // ~O(1)

    // Weights: care most about speed (k90), then ripple, then steps.
    double fitness = 0.5 * k90_norm
                   + 0.3 * ripple_norm
                   + 0.2 * steps_norm;

    return fitness;
}

// ------------------------------------------------------------
// Genetic operators
// ------------------------------------------------------------
DNFParams mutate(const DNFParams& parent, double sigma = 0.1) {
    DNFParams child = parent;
    std::normal_distribution<double> noise(0.0, sigma);

    // perturb a few parameters (small-ish)
    child.tau  += noise(globalRng()) * 0.002;
    child.dt   += noise(globalRng()) * 0.0002;
    child.beta += noise(globalRng()) * 0.2;
    child.h    += noise(globalRng()) * 0.02;
    child.w_exc += noise(globalRng()) * 0.2;
    child.w_inh += noise(globalRng()) * 0.1;
    child.sigma_exc += noise(globalRng()) * 0.3;
    child.sigma_inh += noise(globalRng()) * 0.5;
    child.intensity_scale_factor += noise(globalRng()) * 0.5;

    // discrete params: sometimes flip
    if (randUniform(0.0, 1.0) < 0.3) {
        child.gaussian_stimulus_width += (randUniform(0.0, 1.0) < 0.5 ? -1 : 1);
    }
    if (randUniform(0.0, 1.0) < 0.3) {
        // N ∈ {50, 100, 150, 200}
        const std::vector<int> Ns = {50, 100, 150, 200};
        int idx = std::rand() % Ns.size();
        child.N = Ns[idx];
    }

    // Re-sample until valid
    int tries = 0;
    while (!child.isValid() && tries < 20) {
        child = DNFParams::randomSample();
        ++tries;
    }
    if (!child.isValid()) {
        // fallback, shouldn't really happen
        child = parent;
    }

    return child;
}

DNFParams crossover(const DNFParams& a, const DNFParams& b) {
    DNFParams child = a;
    if (randUniform(0.0, 1.0) < 0.5) child.tau = b.tau;
    if (randUniform(0.0, 1.0) < 0.5) child.dt = b.dt;
    if (randUniform(0.0, 1.0) < 0.5) child.beta = b.beta;
    if (randUniform(0.0, 1.0) < 0.5) child.theta = b.theta;
    if (randUniform(0.0, 1.0) < 0.5) child.h = b.h;
    if (randUniform(0.0, 1.0) < 0.5) child.w_exc = b.w_exc;
    if (randUniform(0.0, 1.0) < 0.5) child.w_inh = b.w_inh;
    if (randUniform(0.0, 1.0) < 0.5) child.sigma_exc = b.sigma_exc;
    if (randUniform(0.0, 1.0) < 0.5) child.sigma_inh = b.sigma_inh;
    if (randUniform(0.0, 1.0) < 0.5) child.intensity_scale_factor = b.intensity_scale_factor;
    if (randUniform(0.0, 1.0) < 0.5) child.gaussian_stimulus_width = b.gaussian_stimulus_width;
    if (randUniform(0.0, 1.0) < 0.5) child.N = b.N;

    if (!child.isValid()) {
        child = DNFParams::randomSample();
    }
    return child;
}

// Tournament selection
int tournamentSelect(const std::vector<double>& fitnesses, int k = 3) {
    std::uniform_int_distribution<int> dist(0, (int)fitnesses.size() - 1);
    int best_idx = dist(globalRng());
    double best_fit = fitnesses[best_idx];

    for (int i = 1; i < k; ++i) {
        int idx = dist(globalRng());
        if (fitnesses[idx] < best_fit) {
            best_fit = fitnesses[idx];
            best_idx = idx;
        }
    }
    return best_idx;
}

// ------------------------------------------------------------
// Main GA loop
// ------------------------------------------------------------
int main() {
    std::cout << "=== Genetic DNF search (multi-metric with gates) ===\n\n";

    const int POP_SIZE = 30;
    const int GENERATIONS = 15;

    std::vector<DNFParams> population(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        population[i] = DNFParams::randomSample();
    }

    double global_best_fitness = std::numeric_limits<double>::infinity();
    Metrics global_best_metrics;
    DNFParams global_best_params;

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        std::cout << "--- Generation " << gen << " ---\n";

        std::vector<double> fitnesses(POP_SIZE);
        std::vector<Metrics> metrics_vec(POP_SIZE);

        // Evaluate population
        for (int i = 0; i < POP_SIZE; ++i) {
            Metrics m;
            if (!evalCandidate(population[i], m)) {
                // If eval fails, give terrible fitness
                fitnesses[i] = 1e9;
            } else {
                fitnesses[i] = computeFitness(m);
                metrics_vec[i] = m;
            }

            std::cout << "  Ind " << i
                      << " -> fitness = " << fitnesses[i]
                      << "  (k90=" << metrics_vec[i].k90_ms
                      << ", k99=" << metrics_vec[i].k99_ms
                      << ", ripple=" << metrics_vec[i].ripple
                      << ", peak=" << metrics_vec[i].peak
                      << ", steps=" << metrics_vec[i].steps
                      << ", pass=" << (metrics_vec[i].pass_gate ? "true" : "false")
                      << ")\n";

            if (fitnesses[i] < global_best_fitness) {
                global_best_fitness = fitnesses[i];
                global_best_metrics = metrics_vec[i];
                global_best_params = population[i];
                std::cout << "    New global best!\n";
            }
        }

        // --- Reproduce (skip after last generation) ---
        if (gen == GENERATIONS - 1) break;

        std::vector<DNFParams> new_pop;
        new_pop.reserve(POP_SIZE);

        // Elitism: keep the current best individual
        int best_idx = 0;
        for (int i = 1; i < POP_SIZE; ++i) {
            if (fitnesses[i] < fitnesses[best_idx]) best_idx = i;
        }
        new_pop.push_back(population[best_idx]);

        // Fill the rest with crossover + mutation
        while ((int)new_pop.size() < POP_SIZE) {
            int p1_idx = tournamentSelect(fitnesses);
            int p2_idx = tournamentSelect(fitnesses);

            DNFParams child = crossover(population[p1_idx], population[p2_idx]);

            if (randUniform(0.0, 1.0) < 0.7) {
                child = mutate(child, 0.5);
            }

            new_pop.push_back(child);
        }

        population = std::move(new_pop);
        std::cout << "\n";
    }

    // --------------------------------------------------------
    // Report final best solution
    // --------------------------------------------------------
    std::cout << "=== Search finished ===\n";
    std::cout << "Best fitness: " << global_best_fitness << "\n";
    std::cout << "Best metrics:\n";
    std::cout << "  k90_ms = " << global_best_metrics.k90_ms << "\n";
    std::cout << "  k99_ms = " << global_best_metrics.k99_ms << "\n";
    std::cout << "  ripple = " << global_best_metrics.ripple << "\n";
    std::cout << "  peak   = " << global_best_metrics.peak << "\n";
    std::cout << "  steps  = " << global_best_metrics.steps << "\n";
    std::cout << "  pass   = " << (global_best_metrics.pass_gate ? "true" : "false") << "\n\n";

    std::cout << "Best params (JSON):\n";
    std::cout << global_best_params.toJSON().dump(2) << "\n";

    std::cout << "=== End of genetic DNF search (multi-metric spec) ===\n";

    return 0;
}
