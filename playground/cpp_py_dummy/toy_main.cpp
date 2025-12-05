#include <iostream>
#include <fstream>
#include <cstdlib>   // for std::system

int main() {
    // 1) Write a simple config file with two numbers: gain and tau
    double gain = 1.5;
    double tau  = 0.8;

    {
        std::ofstream cfg("config.txt");
        if (!cfg) {
            std::cerr << "Failed to open config.txt for writing\n";
            return 1;
        }
        cfg << gain << " " << tau << "\n";
    }

    // 2) Call Python script: python toy_eval.py config.txt
    int ret = std::system("python toy_eval.py config.txt");
    if (ret != 0) {
        std::cerr << "Python script failed with code " << ret << "\n";
        return 1;
    }

    // 3) Read result.txt: "k99_ms ripple peak"
    double k99_ms = 0.0;
    double ripple = 0.0;
    double peak   = 0.0;

    {
        std::ifstream rf("result.txt");
        if (!rf) {
            std::cerr << "Failed to open result.txt for reading\n";
            return 1;
        }
        rf >> k99_ms >> ripple >> peak;
    }

    // 4) Print metrics so we can see it worked
    std::cout << "Got metrics from Python:\n";
    std::cout << "  k99_ms = " << k99_ms << "\n";
    std::cout << "  ripple = " << ripple << "\n";
    std::cout << "  peak   = " << peak   << "\n";

    return 0;
}
