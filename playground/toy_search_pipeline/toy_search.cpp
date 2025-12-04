#include <iostream>
#include <fstream>
#include <cstdlib>    // std::system
#include <limits>
#include <random>
#include <algorithm>
#include <vector>
#include "dnf_params.hpp"

// -----------------------------
// YAML writing + Python eval
// -----------------------------

void write_yaml_config(const DNFParams& p, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open YAML config file: " + path);
    }

    out << "tau: " << p.tau << "\n";
    out << "dt: " << p.dt << "\n";
    out << "beta: " << p.beta << "\n";
    out << "theta: " << p.theta << "\n";
    out << "h: " << p.h << "\n";
    out << "w_exc: " << p.w_exc << "\n";
    out << "sigma_exc: " << p.sigma_exc << "\n";
    out << "w_inh: " << p.w_inh << "\n";
    out << "sigma_inh: " << p.sigma_inh << "\n";
    out << "intensity_scale_factor: " << p.intensity_scale_factor << "\n";
    out << "gaussian_stimulus_width: " << p.gaussian_stimulus_width << "\n";
    out << "N: " << p.N << "\n";
}

// Single-objective evaluation: lower error is better
double evaluate_candidate(const DNFParams& p) {
    const std::string cfg_path = "toy_config.yaml";

    // 1) Write YAML config
    write_yaml_config(p, cfg_path);

    // 2) Call Python evaluator
    int ret = std::system("python toy_eval.py toy_config.yaml");
    if (ret != 0) {
        std::cerr << "Python evaluator failed with code " << ret << "\n";
        return std::numeric_limits<double>::infinity();
    }

    // 3) Read error score from toy_result.txt
    std::ifstream in("toy_result.txt");
    if (!in) {
        std::cerr << "Failed to open toy_result.txt for reading\n";
        return std::numeric_limits<double>::infinity();
    }

    double error = 0.0;
    in >> error;
    return error;
}

// -----------------------------
// Genetic search scaffolding
// -----------------------------

struct Individual {
    DNFParams params;
    double fitness;   // here: error from Python (lower is better)
};

std::mt19937& global_rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// Mutate parameters around their current values while respecting basic ranges.
// If resulting params are invalid, fall back to a fresh random sample.
DNFParams mutate_params(const DNFParams& base, double mutation_scale) {
    auto& gen = global_rng();

    // We'll use different scales per parameter roughly proportional
    // to their allowed ranges.
    std::normal_distribution<double> noise(0.0, mutation_scale);

    DNFParams m = base;

    // tau: 0.005 – 0.020
    m.tau = clamp(m.tau + noise(gen) * (0.020 - 0.005), 0.005, 0.020);

    // dt: 0.0005 – 0.0020
    m.dt = clamp(m.dt + noise(gen) * (0.0020 - 0.0005), 0.0005, 0.0020);

    // beta: 2 – 7
    m.beta = clamp(m.beta + noise(gen) * (7.0 - 2.0), 2.0, 7.0);

    // theta: 0 – 0.1
    m.theta = clamp(m.theta + noise(gen) * 0.1, 0.0, 0.1);

    // h: -0.60 – -0.05
    m.h = clamp(m.h + noise(gen) * (0.60 - 0.05), -0.60, -0.05);

    // w_exc: 0.8 – 3
    m.w_exc = clamp(m.w_exc + noise(gen) * (3.0 - 0.8), 0.8, 3.0);

    // sigma_exc: 2 – 6
    m.sigma_exc = clamp(m.sigma_exc + noise(gen) * (6.0 - 2.0), 2.0, 6.0);

    // w_inh: 0.3 – 2
    m.w_inh = clamp(m.w_inh + noise(gen) * (2.0 - 0.3), 0.3, 2.0);

    // sigma_inh: 3 – 20, sigma_inh > sigma_exc + 1 will be re-checked by isValid()
    m.sigma_inh = clamp(m.sigma_inh + noise(gen) * (20.0 - 3.0), 3.0, 20.0);

    // intensity_scale_factor: 3 – 12
    m.intensity_scale_factor =
        clamp(m.intensity_scale_factor + noise(gen) * (12.0 - 3.0), 3.0, 12.0);

    // gaussian_stimulus_width: 3 – 11 (integers)
    {
        int width = m.gaussian_stimulus_width;
        std::normal_distribution<double> width_noise(0.0, 1.0);
        width = static_cast<int>(std::round(width + width_noise(gen)));
        if (width < 3) width = 3;
        if (width > 11) width = 11;
        m.gaussian_stimulus_width = width;
    }

    // N: one of {50, 100, 150, 200}. Small chance to jump to a neighbor.
    {
        const int valid_Ns[] = {50, 100, 150, 200};
        int current_index = 0;
        for (int i = 0; i < 4; ++i) {
            if (m.N == valid_Ns[i]) {
                current_index = i;
                break;
            }
        }
        std::uniform_int_distribution<int> step_dist(-1, 1);
        int new_index = current_index + step_dist(gen);
        if (new_index < 0) new_index = 0;
        if (new_index > 3) new_index = 3;
        m.N = valid_Ns[new_index];
    }

    // Ensure alpha = dt / tau within (0, 0.2)
    double alpha = m.dt / m.tau;
    if (!(alpha > 0.0 && alpha < 0.2)) {
        // fallback to random
        return DNFParams::randomSample();
    }

    if (!m.isValid()) {
        return DNFParams::randomSample();
    }

    return m;
}

// Simple blend crossover: mix parent params with a random weight.
DNFParams crossover_params(const DNFParams& a, const DNFParams& b) {
    auto& gen = global_rng();
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    double w = wdist(gen);

    DNFParams c;

    c.tau = w * a.tau + (1.0 - w) * b.tau;
    c.dt = w * a.dt + (1.0 - w) * b.dt;
    c.beta = w * a.beta + (1.0 - w) * b.beta;
    c.theta = w * a.theta + (1.0 - w) * b.theta;
    c.h = w * a.h + (1.0 - w) * b.h;
    c.w_exc = w * a.w_exc + (1.0 - w) * b.w_exc;
    c.sigma_exc = w * a.sigma_exc + (1.0 - w) * b.sigma_exc;
    c.w_inh = w * a.w_inh + (1.0 - w) * b.w_inh;
    c.sigma_inh = w * a.sigma_inh + (1.0 - w) * b.sigma_inh;
    c.intensity_scale_factor =
        w * a.intensity_scale_factor + (1.0 - w) * b.intensity_scale_factor;

    // For discrete params, just pick from parents
    c.gaussian_stimulus_width =
        (wdist(gen) < 0.5) ? a.gaussian_stimulus_width : b.gaussian_stimulus_width;
    c.N = (wdist(gen) < 0.5) ? a.N : b.N;

    // If invalid, fallback to random
    if (!c.isValid()) {
        return DNFParams::randomSample();
    }

    return c;
}

// Initialize a random population of Individuals
std::vector<Individual> init_population(int pop_size) {
    std::vector<Individual> pop;
    pop.reserve(pop_size);
    for (int i = 0; i < pop_size; ++i) {
        Individual ind;
        ind.params = DNFParams::randomSample();
        ind.fitness = std::numeric_limits<double>::infinity();
        pop.push_back(ind);
    }
    return pop;
}

// Evaluate every individual in the population
void evaluate_population(std::vector<Individual>& pop) {
    for (auto& ind : pop) {
        ind.fitness = evaluate_candidate(ind.params);
    }
}

// Tournament selection: pick the individual with better fitness
const Individual& tournament_select(const std::vector<Individual>& pop, int tournament_size) {
    auto& gen = global_rng();
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(pop.size()) - 1);

    int best_idx = idx_dist(gen);
    double best_fit = pop[best_idx].fitness;

    for (int i = 1; i < tournament_size; ++i) {
        int j = idx_dist(gen);
        if (pop[j].fitness < best_fit) {
            best_idx = j;
            best_fit = pop[j].fitness;
        }
    }
    return pop[best_idx];
}

// -----------------------------
// "NSGA-II-lite": GA loop
// -----------------------------

void run_genetic_search() {
    const int POP_SIZE = 30;
    const int NUM_GENERATIONS = 15;
    const double CROSSOVER_PROB = 0.9;
    const double MUTATION_SCALE = 0.1; // tweak if needed
    const int TOURNAMENT_SIZE = 3;

    auto& gen = global_rng();
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    std::cout << "=== Genetic search over DNFParams (toy pipeline) ===\n";

    // 1) Initialize population
    std::vector<Individual> population = init_population(POP_SIZE);

    // Track global best
    double best_fitness = std::numeric_limits<double>::infinity();
    DNFParams best_params{};

    for (int g = 0; g < NUM_GENERATIONS; ++g) {
        std::cout << "\n--- Generation " << g << " ---\n";

        // 2) Evaluate population
        evaluate_population(population);

        // 3) Log and update best
        for (int i = 0; i < (int)population.size(); ++i) {
            std::cout << "  Ind " << i
                      << " -> fitness (error) = " << population[i].fitness << "\n";
            if (population[i].fitness < best_fitness) {
                best_fitness = population[i].fitness;
                best_params = population[i].params;
                std::cout << "    New global best!\n";
            }
        }

        // 4) Create next generation
        std::vector<Individual> next_population;
        next_population.reserve(POP_SIZE);

        while ((int)next_population.size() < POP_SIZE) {
            // Parent selection via tournament
            const Individual& p1 = tournament_select(population, TOURNAMENT_SIZE);
            const Individual& p2 = tournament_select(population, TOURNAMENT_SIZE);

            Individual child;
            if (prob_dist(gen) < CROSSOVER_PROB) {
                child.params = crossover_params(p1.params, p2.params);
            } else {
                child.params = p1.params; // clone
            }

            // Mutation
            child.params = mutate_params(child.params, MUTATION_SCALE);
            child.fitness = std::numeric_limits<double>::infinity(); // will be set next generation

            next_population.push_back(child);
        }

        // 5) Replace population
        population = std::move(next_population);
    }

    std::cout << "\n=== Search finished ===\n";
    std::cout << "Best fitness (error): " << best_fitness << "\n";
    auto j = best_params.toJSON();
    std::cout << "Best params (JSON):\n" << j.dump(2) << "\n";
}

int main() {
    run_genetic_search();
    std::cout << "=== End of genetic DNF search ===\n";
    return 0;
}
