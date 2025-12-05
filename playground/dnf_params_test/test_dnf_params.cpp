#include <iostream>
#include "dnf_params.hpp"

int main() {
    std::cout << "=== DNFParams smoke test ===\n";

    // 1) Sample a random valid parameter set
    DNFParams p = DNFParams::randomSample();
    std::cout << "Sampled random DNFParams.\n";

    // 2) Check validity
    bool ok = p.isValid();
    std::cout << "isValid() on sampled params: " << (ok ? "true" : "false") << "\n";

    // 3) Print JSON representation
    auto j = p.toJSON();
    std::cout << "\nJSON dump of sampled params:\n";
    std::cout << j.dump(2) << "\n";

    // 4) Deliberately break a constraint and see isValid() fail
    DNFParams bad = p;
    bad.tau = 0.0001;  // outside [0.005, 0.020]
    bool bad_ok = bad.isValid();
    std::cout << "\nAfter setting tau = 0.0001, isValid(): "
              << (bad_ok ? "true" : "false") << "\n";

    std::cout << "=== End of DNFParams test ===\n";
    return 0;
}
