#pragma once

#include <vector>
#include <cmath>
#include <json.hpp>

struct DNFParams {
    double tau;                      // 0.005 – 0.020
    double dt;                       // 0.0005 – 0.0020
    double beta;                     // 2 – 7
    double theta;                    // 0 – 0.1
    double h;                        // -0.60 – -0.05
    double w_exc;                    // 0.8 – 3
    double sigma_exc;                // 2 – 6
    double w_inh;                   // 0.3 – 2
    double sigma_inh;                // 3 – 20 AND sigma_inh > sigma_exc + 1
    double intensity_scale_factor;   // 3 – 12
    int gaussian_stimulus_width;     // 3 – 11
    int N;                           // must be one of {50, 100, 150, 200}

    bool isValid();
    static DNFParams randomSample();
    nlohmann::json toJSON() const;
};
