#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include "dnf_params.hpp"

namespace nsga2 {

namespace fs = std::filesystem;

struct Individual {
    DNFParams params;
    double obj1 = 0.0; // k99_ms (minimize)
    double obj2 = 0.0; // max_ripple (minimize)
    double obj3 = 0.0; // -min_peak (minimize)
    int rank = 0;
    double crowding = 0.0;
};

struct NSGA2Config {
    int population_size = 50;
    int generations = 50;
    double crossover_rate = 0.9;
    double mutation_rate = 0.1;
    double eta_c = 15.0;
    double eta_m = 20.0;
    unsigned seed = 0;
};

// Run the optimizer and write pareto to `optimizer_results/nsga2/pareto.json`.
void run(const NSGA2Config& cfg);

} // namespace nsga2
