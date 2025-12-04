#include <iostream>
#include "search/nsga2.hpp"

int main() {
    nsga2::NSGA2Config cfg;
    cfg.population_size = 30;
    cfg.generations = 10;
    cfg.crossover_rate = 0.9;
    cfg.mutation_rate = 0.1;

    std::cout << "Starting NSGA-II optimization..." << std::endl;
    nsga2::run(cfg);
    std::cout << "Done. Pareto front saved to optimizer_results/nsga2/pareto.json" << std::endl;
    return 0;
}
