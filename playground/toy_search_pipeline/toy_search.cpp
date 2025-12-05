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
#include "json.hpp"         // nlohmann::json, sitting in the same folder

using nlohmann::json;

// -----------------------------------------------------------------------------
// Config: paths + python command
// -----------------------------------------------------------------------------

// If your toy_eval.py expects different names, just change these 3 strings
static const std::string CONFIG_PATH  = "candidate_dnf_config.yaml";
static const std::string METRICS_PATH = "candidate_metrics.json";
static const std::string PYTHON_CMD_PREFIX = "python toy_eval.py ";

// -----------------------------------------------------------------------------
// Metrics + individuals
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

struct Individual {
    DNFParams params;
    Metrics   metrics;

    // NSGA-II fields
    std::vector<double> objectives;  // generic: any number of objectives
    int    rank        = std::numeric_limits<int>::max();
    double crowd_dist  = 0.0;
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
    // JSON syntax is valid YAML; toy_eval.py can use yaml.safe_load on it.
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

    // Use reasonable defaults if keys missing
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
// Objectives mapping (this is where you encode your spec logic)
// -----------------------------------------------------------------------------

/*
    We treat each thing we care about as a separate objective.
    All objectives are MINIMIZED.

    Example mapping:
      obj0 = k90_ms        (we want response to settle quickly)
      obj1 = ripple        (we want as little ripple as possible)
      obj2 = steps         (we want stability / fewer weird oscillations)
      obj3 = max(0, peak-1)  (penalize overshoot above 1.0)

    Spec failure (metrics.pass == false OR metrics.valid == false):
      -> add a big penalty to every objective so valid configs dominate invalid ones.
*/

void fill_objectives_from_metrics(const Metrics& m, Individual& ind) {
    ind.objectives.clear();

    const double BIG_PENALTY = 1e6;

    // Base objectives (before spec penalty)
    double obj_k90   = m.k90_ms;
    double obj_ripple= m.ripple;
    double obj_steps = m.steps;
    double obj_overshoot = std::max(0.0, m.peak - 1.0);  // 0 if <= 1, positive if overshoot

    ind.objectives.push_back(obj_k90);
    ind.objectives.push_back(obj_ripple);
    ind.objectives.push_back(obj_steps);
    ind.objectives.push_back(obj_overshoot);

    // If eval failed or didn't pass spec, penalize heavily
    if (!m.valid || !m.pass) {
        for (double& v : ind.objectives) {
            v += BIG_PENALTY;
        }
    }
}

// -----------------------------------------------------------------------------
// NSGA-II core: dominance, sorting, crowding distance, selection
// -----------------------------------------------------------------------------

bool dominates(const Individual& a, const Individual& b, double eps = 1e-9) {
    // a dominates b if:
    // - a is no worse than b in ALL objectives
    // - and strictly better in at least one.
    const auto& A = a.objectives;
    const auto& B = b.objectives;

    if (A.size() != B.size()) return false; // shouldn't happen

    bool better_in_any = false;
    for (size_t i = 0; i < A.size(); ++i) {
        double va = A[i];
        double vb = B[i];

        if (va > vb + eps) {
            // a is strictly worse in objective i => cannot dominate
            return false;
        }
        if (va < vb - eps) {
            better_in_any = true;
        }
    }
    return better_in_any;
}

// Fast non-dominated sorting
std::vector<std::vector<int>> fastNonDominatedSort(std::vector<Individual>& pop) {
    size_t N = pop.size();
    std::vector<std::vector<int>> S(N); // S[p] = list of indices dominated by p
    std::vector<int> n_dom(N, 0);       // n_dom[p] = how many individuals dominate p

    std::vector<std::vector<int>> fronts;
    std::vector<int> first_front;

    for (size_t p = 0; p < N; ++p) {
        S[p].clear();
        n_dom[p] = 0;

        for (size_t q = 0; q < N; ++q) {
            if (p == q) continue;
            if (dominates(pop[p], pop[q])) {
                S[p].push_back(static_cast<int>(q));
            } else if (dominates(pop[q], pop[p])) {
                n_dom[p]++;
            }
        }

        if (n_dom[p] == 0) {
            pop[p].rank = 0;
            first_front.push_back(static_cast<int>(p));
        }
    }

    fronts.push_back(first_front);
    int i = 0;
    while (!fronts[i].empty()) {
        std::vector<int> next_front;
        for (int p : fronts[i]) {
            for (int q : S[p]) {
                n_dom[q]--;
                if (n_dom[q] == 0) {
                    pop[q].rank = i + 1;
                    next_front.push_back(q);
                }
            }
        }
        ++i;
        if (!next_front.empty()) {
            fronts.push_back(next_front);
        } else {
            break;
        }
    }

    return fronts;
}

// Crowding distance
void computeCrowdingDistance(std::vector<Individual>& pop,
                             const std::vector<int>& front) {
    const size_t M = front.size();
    if (M == 0) return;

    // Reset crowd distances
    for (int idx : front) {
        pop[idx].crowd_dist = 0.0;
    }

    const size_t n_obj = pop[front[0]].objectives.size();
    if (n_obj == 0) return;

    for (size_t m = 0; m < n_obj; ++m) {
        // Sort front by objective m
        std::vector<int> sorted = front;
        std::sort(sorted.begin(), sorted.end(),
                  [&](int a, int b) {
                      return pop[a].objectives[m] < pop[b].objectives[m];
                  });

        double minv = pop[sorted.front()].objectives[m];
        double maxv = pop[sorted.back()].objectives[m];

        // Extreme points get infinite distance
        pop[sorted.front()].crowd_dist  = std::numeric_limits<double>::infinity();
        pop[sorted.back()].crowd_dist   = std::numeric_limits<double>::infinity();

        if (maxv - minv < 1e-12) {
            // All same value in this objective => no contribution
            continue;
        }

        // Internal points
        for (size_t i = 1; i + 1 < M; ++i) {
            int idx = sorted[i];
            int prev = sorted[i - 1];
            int next = sorted[i + 1];
            double dist = (pop[next].objectives[m] - pop[prev].objectives[m]) / (maxv - minv);
            pop[idx].crowd_dist += dist;
        }
    }
}

// Binary tournament based on rank + crowding distance
int tournamentSelect(const std::vector<Individual>& pop) {
    std::mt19937& gen = globalRNG();
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pop.size()) - 1);

    int i = dist(gen);
    int j = dist(gen);
    while (j == i && pop.size() > 1) {
        j = dist(gen);
    }

    const Individual& a = pop[i];
    const Individual& b = pop[j];

    if (a.rank < b.rank) return i;
    if (b.rank < a.rank) return j;
    // same rank -> higher crowding distance wins
    if (a.crowd_dist > b.crowd_dist) return i;
    if (b.crowd_dist > a.crowd_dist) return j;

    // tie-breaker: random
    return (dist(gen) % 2 == 0) ? i : j;
}

// -----------------------------------------------------------------------------
// Variation: crossover + mutation
// -----------------------------------------------------------------------------

DNFParams crossover(const DNFParams& p1, const DNFParams& p2) {
    std::mt19937& gen = globalRNG();
    std::uniform_real_distribution<double> u(0.0, 1.0);

    DNFParams c;

    auto blend = [&](double a, double b) {
        double t = u(gen);
        return t * a + (1.0 - t) * b;
    };

    c.tau                     = blend(p1.tau,                     p2.tau);
    c.dt                      = blend(p1.dt,                      p2.dt);
    c.beta                    = blend(p1.beta,                    p2.beta);
    c.theta                   = blend(p1.theta,                   p2.theta);
    c.h                       = blend(p1.h,                       p2.h);
    c.w_exc                   = blend(p1.w_exc,                   p2.w_exc);
    c.sigma_exc               = blend(p1.sigma_exc,               p2.sigma_exc);
    c.w_inh                   = blend(p1.w_inh,                   p2.w_inh);
    c.sigma_inh               = blend(p1.sigma_inh,               p2.sigma_inh);
    c.intensity_scale_factor  = blend(p1.intensity_scale_factor,  p2.intensity_scale_factor);

    // For discrete fields, just randomly choose from parents
    c.gaussian_stimulus_width = (u(gen) < 0.5 ? p1.gaussian_stimulus_width
                                              : p2.gaussian_stimulus_width);
    c.N = (u(gen) < 0.5 ? p1.N : p2.N);

    // If somehow invalid, just resample
    if (!c.isValid()) {
        c = DNFParams::randomSample();
    }
    return c;
}

void mutate(DNFParams& p, double mutation_prob = 0.3) {
    std::mt19937& gen = globalRNG();
    std::uniform_real_distribution<double> u(0.0, 1.0);

    // Very simple mutation: with some probability, resample completely.
    if (u(gen) < mutation_prob) {
        p = DNFParams::randomSample();
    }
    // otherwise leave as is
}

// Evaluate one individual and fill metrics + objectives
void evaluateIndividual(Individual& ind) {
    ind.metrics = run_python_eval(ind.params);
    fill_objectives_from_metrics(ind.metrics, ind);
}

// -----------------------------------------------------------------------------
// Utility: pick a representative "best" from the first Pareto front
// -----------------------------------------------------------------------------

int pickRepresentative(const std::vector<Individual>& pop,
                       const std::vector<int>& front) {
    if (front.empty()) return -1;

    int best_idx = front[0];
    double best_sum = std::numeric_limits<double>::infinity();

    for (int idx : front) {
        const Individual& ind = pop[idx];
        double sum = std::accumulate(ind.objectives.begin(),
                                     ind.objectives.end(), 0.0);
        if (sum < best_sum) {
            best_sum = sum;
            best_idx = idx;
        }
    }
    return best_idx;
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

int main() {
    std::cout << "=== NSGA-II style multi-objective DNF search (toy pipeline) ===\n\n";

    const int POP_SIZE   = 30;
    const int N_GEN      = 15;

    // ----------------------------
    // 1) Initialize population
    // ----------------------------
    std::vector<Individual> population(POP_SIZE);
    for (auto& ind : population) {
        ind.params = DNFParams::randomSample();
        evaluateIndividual(ind);
    }

    // Initial fronts + crowding for selection
    auto fronts = fastNonDominatedSort(population);
    for (const auto& f : fronts) {
        computeCrowdingDistance(population, f);
    }

    // ----------------------------
    // 2) Evolution loop
    // ----------------------------
    for (int gen = 0; gen < N_GEN; ++gen) {
        std::cout << "--- Generation " << gen << " ---\n";

        // A bit of logging: how many pass spec, what's the best (sum of objectives)
        int pass_count = 0;
        double best_sum = std::numeric_limits<double>::infinity();
        for (const auto& ind : population) {
            if (ind.metrics.valid && ind.metrics.pass) pass_count++;
            double sum = std::accumulate(ind.objectives.begin(),
                                         ind.objectives.end(), 0.0);
            best_sum = std::min(best_sum, sum);
        }
        std::cout << "  Pass count: " << pass_count << " / " << POP_SIZE
                  << ", approx best scalarized obj sum = " << best_sum << "\n";

        // Create offspring via tournament selection + crossover + mutation
        std::vector<Individual> offspring;
        offspring.reserve(POP_SIZE);

        while (static_cast<int>(offspring.size()) < POP_SIZE) {
            int idx1 = tournamentSelect(population);
            int idx2 = tournamentSelect(population);

            const Individual& p1 = population[idx1];
            const Individual& p2 = population[idx2];

            Individual child1;
            Individual child2;

            child1.params = crossover(p1.params, p2.params);
            child2.params = crossover(p1.params, p2.params);

            mutate(child1.params);
            mutate(child2.params);

            evaluateIndividual(child1);
            evaluateIndividual(child2);

            offspring.push_back(child1);
            if (static_cast<int>(offspring.size()) < POP_SIZE) {
                offspring.push_back(child2);
            }
        }

        // Combine parent + offspring
        std::vector<Individual> combined;
        combined.reserve(population.size() + offspring.size());
        combined.insert(combined.end(), population.begin(), population.end());
        combined.insert(combined.end(), offspring.begin(), offspring.end());

        // Non-dominated sort on combined population
        auto combined_fronts = fastNonDominatedSort(combined);
        for (const auto& f : combined_fronts) {
            computeCrowdingDistance(combined, f);
        }

        // Build next population from fronts
        std::vector<Individual> next_pop;
        next_pop.reserve(POP_SIZE);

        for (size_t f_idx = 0; f_idx < combined_fronts.size(); ++f_idx) {
            const auto& front = combined_fronts[f_idx];
            if (next_pop.size() + front.size() <= static_cast<size_t>(POP_SIZE)) {
                // take entire front
                for (int idx : front) {
                    next_pop.push_back(combined[idx]);
                }
            } else {
                // Need only a part of this front -> sort by crowding distance (desc)
                std::vector<int> sorted = front;
                std::sort(sorted.begin(), sorted.end(),
                          [&](int a, int b) {
                              return combined[a].crowd_dist > combined[b].crowd_dist;
                          });
                size_t remaining = POP_SIZE - next_pop.size();
                for (size_t k = 0; k < remaining && k < sorted.size(); ++k) {
                    next_pop.push_back(combined[sorted[k]]);
                }
                break;
            }
        }

        population = std::move(next_pop);
        // recompute fronts + crowding for next gen selection
        fronts = fastNonDominatedSort(population);
        for (const auto& f : fronts) {
            computeCrowdingDistance(population, f);
        }

        std::cout << "\n";
    }

    // ----------------------------
    // 3) Final result: take first Pareto front and pick representative
    // ----------------------------
    auto final_fronts = fastNonDominatedSort(population);
    if (final_fronts.empty() || final_fronts[0].empty()) {
        std::cerr << "[ERROR] No individuals in final Pareto front? Something went wrong.\n";
        return 1;
    }

    int rep_idx = pickRepresentative(population, final_fronts[0]);
    if (rep_idx < 0) {
        std::cerr << "[ERROR] Could not pick representative individual.\n";
        return 1;
    }

    const Individual& best = population[rep_idx];
    double best_sum = std::accumulate(best.objectives.begin(),
                                      best.objectives.end(), 0.0);

    std::cout << "=== Search finished (NSGA-II style) ===\n";
    std::cout << "Representative 'best' scalarized objective sum: " << best_sum << "\n";
    std::cout << "Best metrics:\n";
    std::cout << "  k90_ms = " << best.metrics.k90_ms << "\n";
    std::cout << "  k99_ms = " << best.metrics.k99_ms << "\n";
    std::cout << "  ripple = " << best.metrics.ripple << "\n";
    std::cout << "  peak   = " << best.metrics.peak << "\n";
    std::cout << "  steps  = " << best.metrics.steps << "\n";
    std::cout << "  pass   = " << (best.metrics.pass ? "true" : "false") << "\n\n";

    std::cout << "Best params (JSON):\n";
    std::cout << best.params.toJSON().dump(2) << "\n";
    std::cout << "=== End of NSGA-II multi-objective DNF search ===\n";

    return 0;
}
