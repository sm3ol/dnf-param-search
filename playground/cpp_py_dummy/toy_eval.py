# toy_eval.py
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python toy_eval.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    # 1) read simple text config: "gain tau"
    with open(config_path, "r") as f:
        line = f.read().strip()
    parts = line.split()
    gain = float(parts[0])
    tau = float(parts[1])

    # 2) fake "metrics" computed from those values
    k99_ms = gain * 10.0            # just some made-up formula
    ripple = 1.0 / (tau + 1e-3)     # smaller tau => larger ripple
    peak   = gain * tau

    # 3) write them out as plain text: "k99_ms ripple peak"
    with open("result.txt", "w") as f:
        f.write(f"{k99_ms} {ripple} {peak}\n")

if __name__ == "__main__":
    main()
