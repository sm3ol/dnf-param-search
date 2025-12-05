#include "dnf_params.hpp"

#include <random>
#include <limits>
#include <algorithm>

using nlohmann::json;

bool DNFParams::isValid() {
    // Basic finite checks
    auto finite = [](double v){ return std::isfinite(v); };
    if (!finite(tau) || !finite(dt) || !finite(beta) || !finite(theta) || !finite(h) ||
        !finite(w_exc) || !finite(sigma_exc) || !finite(w_inh) || !finite(sigma_inh) ||
        !finite(intensity_scale_factor)) return false;

    if (tau < 0.005 || tau > 0.020) return false;
    if (dt < 0.0005 || dt > 0.0020) return false;

    double alpha = dt / tau;
    if (!(alpha > 0.0 && alpha < 0.2)) return false;

    if (beta < 2.0 || beta > 7.0) return false;
    if (theta < 0.0 || theta > 0.1) return false;
    if (h < -0.60 || h > -0.05) return false;
    if (w_exc < 0.8 || w_exc > 3.0) return false;
    if (sigma_exc < 2.0 || sigma_exc > 6.0) return false;
    if (w_inh < 0.3 || w_inh > 2.0) return false;
    if (sigma_inh < 3.0 || sigma_inh > 20.0) return false;
    if (!(sigma_inh > sigma_exc + 1.0)) return false;
    if (intensity_scale_factor < 3.0 || intensity_scale_factor > 12.0) return false;
    if (gaussian_stimulus_width < 3 || gaussian_stimulus_width > 11) return false;

    if (!(N == 50 || N == 100 || N == 150 || N == 200)) return false;

    return true;
}

DNFParams DNFParams::randomSample() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> tau_dist(0.005, 0.020);
    std::uniform_real_distribution<double> dt_dist(0.0005, 0.0020);
    std::uniform_real_distribution<double> beta_dist(2.0, 7.0);
    std::uniform_real_distribution<double> theta_dist(0.0, 0.1);
    std::uniform_real_distribution<double> h_dist(-0.60, -0.05);
    std::uniform_real_distribution<double> w_exc_dist(0.8, 3.0);
    std::uniform_real_distribution<double> sigma_exc_dist(2.0, 6.0);
    std::uniform_real_distribution<double> w_inh_dist(0.3, 2.0);
    std::uniform_real_distribution<double> intensity_dist(3.0, 12.0);
    std::uniform_int_distribution<int> gauss_width_dist(3, 11);

    const std::vector<int> N_options = {50, 100, 150, 200};
    std::uniform_int_distribution<int> N_index(0, static_cast<int>(N_options.size()) - 1);

    while (true) {
        DNFParams p;
        p.tau = tau_dist(gen);

        // Sample dt but enforce alpha = dt/tau in (0, 0.2)
        bool dt_ok = false;
        for (int tries = 0; tries < 1000; ++tries) {
            p.dt = dt_dist(gen);
            double alpha = p.dt / p.tau;
            if (alpha > 0.0 && alpha < 0.2) { dt_ok = true; break; }
        }
        if (!dt_ok) continue; // rare, but resample

        p.beta = beta_dist(gen);
        p.theta = theta_dist(gen);
        p.h = h_dist(gen);
        p.w_exc = w_exc_dist(gen);
        p.sigma_exc = sigma_exc_dist(gen);
        p.w_inh = w_inh_dist(gen);

        // Ensure sigma_inh > sigma_exc + 1 and within [3,20]
        double sigma_inh_min = std::max(3.0, p.sigma_exc + 1.0000001);
        if (sigma_inh_min > 20.0) continue; // cannot satisfy, resample
        std::uniform_real_distribution<double> sigma_inh_dist(sigma_inh_min, 20.0);
        p.sigma_inh = sigma_inh_dist(gen);

        p.intensity_scale_factor = intensity_dist(gen);
        p.gaussian_stimulus_width = gauss_width_dist(gen);
        p.N = N_options[N_index(gen)];

        if (p.isValid()) return p;
        // otherwise loop and resample
    }
}

json DNFParams::toJSON() const {
    json j;
    j["tau"] = tau;
    j["dt"] = dt;
    j["beta"] = beta;
    j["theta"] = theta;
    j["h"] = h;
    j["w_exc"] = w_exc;
    j["sigma_exc"] = sigma_exc;
    j["w_inh"] = w_inh;
    j["sigma_inh"] = sigma_inh;
    j["intensity_scale_factor"] = intensity_scale_factor;
    j["gaussian_stimulus_width"] = gaussian_stimulus_width;
    j["N"] = N;
    return j;
}
