import sys
import math
import yaml  # you'll need PyYAML installed: pip install pyyaml

# Define the "true" / ideal parameter set in this toy world
TARGET = {
    "tau": 0.015,
    "dt": 0.001,
    "beta": 5.0,
    "theta": 0.05,
    "h": -0.3,
    "w_exc": 2.0,
    "sigma_exc": 4.0,
    "w_inh": 1.0,
    "sigma_inh": 8.0,
    "intensity_scale_factor": 6.0,
    "gaussian_stimulus_width": 7,
    "N": 100,
}

def main():
    if len(sys.argv) != 2:
        print("Usage: python toy_eval.py <config_yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    # 1. Load YAML config written by C++
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Compute a simple error score: sum of squared differences
    #    For ints we still just treat them as numbers.
    error = 0.0
    for key, target_val in TARGET.items():
        cand_val = cfg.get(key)
        if cand_val is None:
            # Penalize missing keys heavily
            error += 1000.0
            continue
        diff = float(cand_val) - float(target_val)
        error += diff * diff

    # 3. Write result to a text file so C++ can read it easily
    with open("toy_result.txt", "w") as f:
        f.write(f"{error}\n")

if __name__ == "__main__":
    main()
