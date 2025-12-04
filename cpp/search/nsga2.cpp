#include "nsga2.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <chrono>

namespace nsga2 {

using json = nlohmann::json;
namespace fs = std::filesystem;

static std::mt19937& globalRng(unsigned seed = 0) {
    static std::mt19937 rng(std::random_device{}());
    if (seed != 0) rng.seed(seed);
    return rng;
}

// Bounds and helpers (match DNFParams::isValid)
struct Bounds {
    static constexpr double tau_min = 0.005, tau_max = 0.020;
    static constexpr double dt_min = 0.0005, dt_max = 0.0020;
    static constexpr double beta_min = 2.0, beta_max = 7.0;
    static constexpr double theta_min = 0.0, theta_max = 0.1;
    static constexpr double h_min = -0.60, h_max = -0.05;
    static constexpr double w_exc_min = 0.8, w_exc_max = 3.0;
    static constexpr double sigma_exc_min = 2.0, sigma_exc_max = 6.0;
    static constexpr double w_inh_min = 0.3, w_inh_max = 2.0;
    static constexpr double sigma_inh_min = 3.0, sigma_inh_max = 20.0;
    static constexpr double intensity_min = 3.0, intensity_max = 12.0;
    static constexpr int gauss_min = 3, gauss_max = 11;
    static const std::vector<int>& N_options() {
        static std::vector<int> opt{50,100,150,200};
        return opt;
    }
};

static double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
static int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Write temporary YAML with params (minimal form expected by the black-box)
static bool writeTempYAML(const DNFParams& p, const fs::path& path) {
    std::ofstream os(path);
    if (!os) return false;
    os << "params:\n";
    os << "  tau: " << p.tau << "\n";
    os << "  dt: " << p.dt << "\n";
    os << "  beta: " << p.beta << "\n";
    os << "  theta: " << p.theta << "\n";
    os << "  h: " << p.h << "\n";
    os << "  w_exc: " << p.w_exc << "\n";
    os << "  sigma_exc: " << p.sigma_exc << "\n";
    os << "  w_inh: " << p.w_inh << "\n";
    os << "  sigma_inh: " << p.sigma_inh << "\n";
    os << "  intensity_scale_factor: " << p.intensity_scale_factor << "\n";
    os << "  gaussian_stimulus_width: " << p.gaussian_stimulus_width << "\n";
    os << "  N: " << p.N << "\n";
    os.close();
    return true;
}

// Parse registry.jsonl and extract metrics. Returns true on success.
static bool parseRegistry(const fs::path& registryPath, double& k99_ms, double& max_ripple, double& min_peak) {
    std::ifstream is(registryPath);
    if (!is) return false;
    std::string line;
    json j;
    // read last non-empty line
    std::string last;
    while (std::getline(is, line)) {
        if (line.size() && line.find_first_not_of(" \t\r\n") != std::string::npos) last = line;
    }
    if (last.empty()) return false;
    try {
        j = json::parse(last);
    } catch (...) { return false; }

    // tolerant extraction
    if (j.contains("k99_ms")) k99_ms = j["k99_ms"].get<double>(); else return false;
    if (j.contains("max_ripple")) max_ripple = j["max_ripple"].get<double>(); else return false;
    if (j.contains("min_peak")) min_peak = j["min_peak"].get<double>(); else return false;
    return true;
}

// Run the external python black-box using the exact command given by the user
static int runBlackBox() {
    const char* cmd = "python -m python_env.bb_run --config temp.yaml --cert configs/certification.yaml --datasets configs/datasets.yaml --dataset ycbsight_real --split test --out artifacts/runs/temp --max-frames 80 --per-sample --sample-steps 200 --quiet";
    return std::system(cmd);
}

// Evaluate an individual: write YAML, remove previous output, run black-box, parse registry.jsonl
static void evaluateIndividual(Individual& ind) {
    // prepare temp.yaml in current working directory
    fs::path tempYaml = fs::path("temp.yaml");
    if (!writeTempYAML(ind.params, tempYaml)) {
        ind.obj1 = ind.obj2 = ind.obj3 = std::numeric_limits<double>::infinity();
        return;
    }

    // remove previous output dir if present
    fs::path outdir = fs::path("artifacts") / "runs" / "temp";
    std::error_code ec;
    fs::remove_all(outdir, ec);

    int rc = runBlackBox();
    if (rc != 0) {
        ind.obj1 = ind.obj2 = ind.obj3 = std::numeric_limits<double>::infinity();
        return;
    }

    // parse registry
    fs::path registry = outdir / "registry.jsonl";
    double k99_ms=0, max_ripple=0, min_peak=0;
    bool ok = parseRegistry(registry, k99_ms, max_ripple, min_peak);
    if (!ok) {
        ind.obj1 = ind.obj2 = ind.obj3 = std::numeric_limits<double>::infinity();
        return;
    }

    ind.obj1 = k99_ms;
    ind.obj2 = max_ripple;
    ind.obj3 = -min_peak; // convert maximize->minimize
}

// Dominance: returns true if a dominates b (minimization)
static bool dominates(const Individual& a, const Individual& b) {
    bool better = false;
    if (a.obj1 > b.obj1) return false;
    if (a.obj2 > b.obj2) return false;
    if (a.obj3 > b.obj3) return false;
    if (a.obj1 < b.obj1) better = true;
    if (a.obj2 < b.obj2) better = true;
    if (a.obj3 < b.obj3) better = true;
    return better;
}

// Fast non-dominated sort
static std::vector<std::vector<int>> fastNonDominatedSort(std::vector<Individual>& pop) {
    int N = (int)pop.size();
    std::vector<std::vector<int>> S(N);
    std::vector<int> n(N,0);
    std::vector<std::vector<int>> fronts;
    fronts.emplace_back();

    for (int p=0;p<N;++p) {
        for (int q=0;q<N;++q) {
            if (p==q) continue;
            if (dominates(pop[p], pop[q])) S[p].push_back(q);
            else if (dominates(pop[q], pop[p])) n[p]++;
        }
        if (n[p]==0) { pop[p].rank = 1; fronts[0].push_back(p); }
    }

    int i = 0;
    while (!fronts[i].empty()) {
        std::vector<int> Q;
        for (int p : fronts[i]) {
            for (int q : S[p]) {
                n[q]--;
                if (n[q]==0) {
                    pop[q].rank = i+2;
                    Q.push_back(q);
                }
            }
        }
        ++i;
        if (!Q.empty()) fronts.push_back(Q);
        else break;
    }
    return fronts;
}

// Crowding distance for a front (indices into population)
static void crowdingDistance(std::vector<Individual>& pop, const std::vector<int>& front) {
    int l = (int)front.size();
    if (l==0) return;
    for (int idx : front) pop[idx].crowding = 0.0;
    // For each objective
    for (int obj=0; obj<3; ++obj) {
        std::vector<int> ord = front;
        std::sort(ord.begin(), ord.end(), [&](int a, int b){
            if (obj==0) return pop[a].obj1 < pop[b].obj1;
            if (obj==1) return pop[a].obj2 < pop[b].obj2;
            return pop[a].obj3 < pop[b].obj3;
        });
        // assign infinity for boundaries
        pop[ord.front()].crowding = std::numeric_limits<double>::infinity();
        pop[ord.back()].crowding = std::numeric_limits<double>::infinity();
        double fmin, fmax;
        if (obj==0) { fmin = pop[ord.front()].obj1; fmax = pop[ord.back()].obj1; }
        else if (obj==1) { fmin = pop[ord.front()].obj2; fmax = pop[ord.back()].obj2; }
        else { fmin = pop[ord.front()].obj3; fmax = pop[ord.back()].obj3; }
        double denom = fmax - fmin;
        if (denom == 0.0) denom = 1.0; // avoid div by zero
        for (int k=1;k<l-1;++k) {
            double prev, next;
            if (obj==0) { prev = pop[ord[k-1]].obj1; next = pop[ord[k+1]].obj1; }
            else if (obj==1) { prev = pop[ord[k-1]].obj2; next = pop[ord[k+1]].obj2; }
            else { prev = pop[ord[k-1]].obj3; next = pop[ord[k+1]].obj3; }
            pop[ord[k]].crowding += (next - prev) / denom;
        }
    }
}

// Tournament selection (binary)
static const Individual& tournamentSelect(const std::vector<Individual>& pop, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, (int)pop.size()-1);
    int a = dist(rng);
    int b = dist(rng);
    const Individual& A = pop[a];
    const Individual& B = pop[b];
    if (A.rank < B.rank) return A;
    if (B.rank < A.rank) return B;
    // equal rank -> prefer larger crowding
    if (A.crowding > B.crowding) return A;
    return B;
}

// Repair to satisfy bounds and simple relational constraints
static void repairParams(DNFParams& p) {
    p.tau = clamp(p.tau, Bounds::tau_min, Bounds::tau_max);
    p.dt = clamp(p.dt, Bounds::dt_min, Bounds::dt_max);
    // ensure alpha in (0, 0.2) by clamping dt relative to tau
    double alpha = p.dt / p.tau;
    if (alpha <= 0.0) p.dt = std::max(1e-6, p.tau * 1e-3);
    if (alpha >= 0.2) p.dt = p.tau * 0.1999;

    p.beta = clamp(p.beta, Bounds::beta_min, Bounds::beta_max);
    p.theta = clamp(p.theta, Bounds::theta_min, Bounds::theta_max);
    p.h = clamp(p.h, Bounds::h_min, Bounds::h_max);
    p.w_exc = clamp(p.w_exc, Bounds::w_exc_min, Bounds::w_exc_max);
    p.sigma_exc = clamp(p.sigma_exc, Bounds::sigma_exc_min, Bounds::sigma_exc_max);
    p.w_inh = clamp(p.w_inh, Bounds::w_inh_min, Bounds::w_inh_max);

    // ensure sigma_inh > sigma_exc + 1.0 and within [3,20]
    double min_sigma_inh = std::max(Bounds::sigma_inh_min, p.sigma_exc + 1.0000001);
    p.sigma_inh = clamp(p.sigma_inh, min_sigma_inh, Bounds::sigma_inh_max);

    p.intensity_scale_factor = clamp(p.intensity_scale_factor, Bounds::intensity_min, Bounds::intensity_max);
    p.gaussian_stimulus_width = clampInt(p.gaussian_stimulus_width, Bounds::gauss_min, Bounds::gauss_max);
    // clamp N to nearest available option
    const auto& opts = Bounds::N_options();
    int best = opts.front();
    int bestd = std::abs(p.N - best);
    for (int v : opts) { int d = std::abs(p.N - v); if (d < bestd) { bestd = d; best = v; } }
    p.N = best;
}

// SBX Crossover for two parents -> two children (on DNFParams)
static void sbxCrossover(const DNFParams& p1, const DNFParams& p2, DNFParams& c1, DNFParams& c2, double pc, double eta_c, std::mt19937& rng) {
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    auto sbx = [&](double x1, double x2, double lo, double hi) -> std::pair<double,double> {
        if (ur(rng) > pc) return {x1, x2};
        if (std::abs(x1 - x2) < 1e-12) return {x1, x2};
        double x_low = lo;
        double x_high = hi;
        double y1 = std::min(x1, x2);
        double y2 = std::max(x1, x2);
        double rand = ur(rng);
        double beta = 1.0 + (2.0*(y1 - x_low)/(y2 - y1));
        double alpha = 2.0 - std::pow(beta, -(eta_c+1.0));
        double betaq;
        if (rand <= 1.0/alpha) betaq = std::pow(rand*alpha, 1.0/(eta_c+1.0));
        else betaq = std::pow(1.0/(2.0 - rand*alpha), 1.0/(eta_c+1.0));
        double child1 = 0.5*((y1 + y2) - betaq*(y2 - y1));

        beta = 1.0 + (2.0*(x_high - y2)/(y2 - y1));
        alpha = 2.0 - std::pow(beta, -(eta_c+1.0));
        rand = ur(rng);
        if (rand <= 1.0/alpha) betaq = std::pow(rand*alpha, 1.0/(eta_c+1.0));
        else betaq = std::pow(1.0/(2.0 - rand*alpha), 1.0/(eta_c+1.0));
        double child2 = 0.5*((y1 + y2) + betaq*(y2 - y1));

        // clamp
        child1 = clamp(child1, lo, hi);
        child2 = clamp(child2, lo, hi);
        if (ur(rng) < 0.5) return {child1, child2};
        else return {child2, child1};
    };

    // continuous fields
    auto pr = sbx(p1.tau, p2.tau, Bounds::tau_min, Bounds::tau_max); c1.tau = pr.first; c2.tau = pr.second;
    pr = sbx(p1.dt, p2.dt, Bounds::dt_min, Bounds::dt_max); c1.dt = pr.first; c2.dt = pr.second;
    pr = sbx(p1.beta, p2.beta, Bounds::beta_min, Bounds::beta_max); c1.beta = pr.first; c2.beta = pr.second;
    pr = sbx(p1.theta, p2.theta, Bounds::theta_min, Bounds::theta_max); c1.theta = pr.first; c2.theta = pr.second;
    pr = sbx(p1.h, p2.h, Bounds::h_min, Bounds::h_max); c1.h = pr.first; c2.h = pr.second;
    pr = sbx(p1.w_exc, p2.w_exc, Bounds::w_exc_min, Bounds::w_exc_max); c1.w_exc = pr.first; c2.w_exc = pr.second;
    pr = sbx(p1.sigma_exc, p2.sigma_exc, Bounds::sigma_exc_min, Bounds::sigma_exc_max); c1.sigma_exc = pr.first; c2.sigma_exc = pr.second;
    pr = sbx(p1.w_inh, p2.w_inh, Bounds::w_inh_min, Bounds::w_inh_max); c1.w_inh = pr.first; c2.w_inh = pr.second;
    // sigma_inh uses min determined by sigma_exc + 1 for each child
    double s_inh_lo1 = std::max(Bounds::sigma_inh_min, c1.sigma_exc + 1.0000001);
    double s_inh_lo2 = std::max(Bounds::sigma_inh_min, c2.sigma_exc + 1.0000001);
    pr = sbx(p1.sigma_inh, p2.sigma_inh, s_inh_lo1, Bounds::sigma_inh_max); c1.sigma_inh = pr.first;
    pr = sbx(p1.sigma_inh, p2.sigma_inh, s_inh_lo2, Bounds::sigma_inh_max); c2.sigma_inh = pr.second;
    pr = sbx(p1.intensity_scale_factor, p2.intensity_scale_factor, Bounds::intensity_min, Bounds::intensity_max);
    c1.intensity_scale_factor = pr.first; c2.intensity_scale_factor = pr.second;

    // discrete fields: gaussian width
    if (ur(rng) <= 0.5) { c1.gaussian_stimulus_width = p1.gaussian_stimulus_width; c2.gaussian_stimulus_width = p2.gaussian_stimulus_width; }
    else { c1.gaussian_stimulus_width = p2.gaussian_stimulus_width; c2.gaussian_stimulus_width = p1.gaussian_stimulus_width; }

    // N: choose from parents or random nearby
    if (ur(rng) <= 0.5) { c1.N = p1.N; c2.N = p2.N; }
    else { c1.N = p2.N; c2.N = p1.N; }

    // final repair
    repairParams(c1);
    repairParams(c2);
}

// Polynomial mutation
static void polynomialMutate(DNFParams& p, double pm, double eta_m, std::mt19937& rng) {
    std::uniform_real_distribution<double> ur(0.0,1.0);
    // mutate continuous
    auto mutate = [&](double& x, double lo, double hi) {
        if (ur(rng) > pm) return;
        double u = ur(rng);
        double delta1 = (x - lo) / (hi - lo);
        double delta2 = (hi - x) / (hi - lo);
        double mut_pow = 1.0 / (eta_m + 1.0);
        double deltaq;
        if (u <= 0.5) {
            double xy = 1.0 - delta1;
            double val = 2.0*u + (1.0-2.0*u)*std::pow(xy, (eta_m+1.0));
            deltaq = std::pow(val, mut_pow) - 1.0;
        } else {
            double xy = 1.0 - delta2;
            double val = 2.0*(1.0-u) + 2.0*(u-0.5)*std::pow(xy, (eta_m+1.0));
            deltaq = 1.0 - std::pow(val, mut_pow);
        }
        x = x + deltaq * (hi - lo);
        x = clamp(x, lo, hi);
    };

    mutate(p.tau, Bounds::tau_min, Bounds::tau_max);
    mutate(p.dt, Bounds::dt_min, Bounds::dt_max);
    mutate(p.beta, Bounds::beta_min, Bounds::beta_max);
    mutate(p.theta, Bounds::theta_min, Bounds::theta_max);
    mutate(p.h, Bounds::h_min, Bounds::h_max);
    mutate(p.w_exc, Bounds::w_exc_min, Bounds::w_exc_max);
    mutate(p.sigma_exc, Bounds::sigma_exc_min, Bounds::sigma_exc_max);
    mutate(p.w_inh, Bounds::w_inh_min, Bounds::w_inh_max);
    mutate(p.sigma_inh, Bounds::sigma_inh_min, Bounds::sigma_inh_max);
    mutate(p.intensity_scale_factor, Bounds::intensity_min, Bounds::intensity_max);

    // discrete mutations
    if (ur(rng) < pm) {
        std::uniform_int_distribution<int> gdist(Bounds::gauss_min, Bounds::gauss_max);
        p.gaussian_stimulus_width = gdist(rng);
    }
    if (ur(rng) < pm) {
        const auto& opts = Bounds::N_options();
        std::uniform_int_distribution<int> ndist(0, (int)opts.size()-1);
        p.N = opts[ndist(rng)];
    }

    repairParams(p);
}

// Environmental selection: sort by rank then crowding and keep best N
static std::vector<Individual> environmentalSelection(std::vector<Individual>& combined, int N) {
    std::vector<Individual> out;
    auto fronts = fastNonDominatedSort(combined);
    for (const auto& f : fronts) {
        // compute crowding for this front
        crowdingDistance(combined, f);
        // collect sorted by crowding descending
        std::vector<int> sorted = f;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b){
            return combined[a].crowding > combined[b].crowding;
        });
        for (int idx : sorted) {
            if ((int)out.size() < N) out.push_back(combined[idx]);
        }
        if ((int)out.size() >= N) break;
    }
    // if not enough (shouldn't happen), pad randomly
    while ((int)out.size() < N) out.push_back(combined[std::uniform_int_distribution<int>(0,(int)combined.size()-1)(globalRng())]);
    return out;
}

// Main run
void run(const NSGA2Config& cfg) {
    std::mt19937& rng = globalRng(cfg.seed);
    int popSize = cfg.population_size;

    // initialize population
    std::vector<Individual> pop(popSize);
    for (int i=0;i<popSize;++i) {
        pop[i].params = DNFParams::randomSample();
        // set placeholder objectives large until evaluated
        pop[i].obj1 = pop[i].obj2 = pop[i].obj3 = std::numeric_limits<double>::infinity();
    }

    // evaluate initial population
    for (int i=0;i<popSize;++i) evaluateIndividual(pop[i]);

    for (int gen=0; gen<cfg.generations; ++gen) {
        std::vector<Individual> offspring;
        offspring.reserve(popSize);
        // prepare selection: compute ranks and crowding distances
        auto fronts = fastNonDominatedSort(pop);
        for (const auto& f : fronts) crowdingDistance(pop, f);

        // generate offspring
        while ((int)offspring.size() < popSize) {
            const Individual& parent1 = tournamentSelect(pop, rng);
            const Individual& parent2 = tournamentSelect(pop, rng);
            Individual child1, child2;
            // SBX
            sbxCrossover(parent1.params, parent2.params, child1.params, child2.params, cfg.crossover_rate, cfg.eta_c, rng);
            // mutation
            polynomialMutate(child1.params, cfg.mutation_rate, cfg.eta_m, rng);
            polynomialMutate(child2.params, cfg.mutation_rate, cfg.eta_m, rng);
            // evaluate
            evaluateIndividual(child1);
            if ((int)offspring.size() + 1 < popSize) evaluateIndividual(child2);
            offspring.push_back(std::move(child1));
            if ((int)offspring.size() < popSize) offspring.push_back(std::move(child2));
        }

        // combine and select next generation
        std::vector<Individual> combined = pop;
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        pop = environmentalSelection(combined, popSize);

        // progress report to stdout
        std::cout << "Generation " << (gen+1) << " completed" << std::endl;
    }

    // compute final fronts and collect Pareto (rank 1)
    auto fronts = fastNonDominatedSort(pop);
    std::vector<Individual> pareto;
    if (!fronts.empty()) {
        for (int idx : fronts[0]) pareto.push_back(pop[idx]);
    }

    // ensure output dir
    fs::path outdir = fs::path("optimizer_results") / "nsga2";
    std::error_code ec;
    fs::create_directories(outdir, ec);

    // write pareto JSON
    fs::path outFile = outdir / "pareto.json";
    json jout = json::array();
    for (const auto& ind : pareto) {
        json e = ind.params.toJSON();
        e["obj_k99_ms"] = ind.obj1;
        e["obj_max_ripple"] = ind.obj2;
        e["obj_neg_min_peak"] = ind.obj3;
        e["rank"] = ind.rank;
        e["crowding"] = ind.crowding;
        jout.push_back(e);
    }
    std::ofstream os(outFile);
    if (os) os << jout.dump(2);
    else std::cerr << "Failed writing pareto file: " << outFile.string() << std::endl;
}

} // namespace nsga2
