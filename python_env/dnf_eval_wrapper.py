#!/usr/bin/env python3
import sys
import json
import subprocess
from pathlib import Path
import csv
import time


RUNS_ROOT = Path("artifacts") / "runs"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def find_latest_run_dir():
    if not RUNS_ROOT.exists():
        raise RuntimeError(f"Runs root {RUNS_ROOT} does not exist")
    # Take the most recently modified directory
    dirs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise RuntimeError(f"No run directories found under {RUNS_ROOT}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def read_metrics_from_csv(run_dir: Path):
    # There should be exactly one *_samples.csv in the run dir
    csv_files = list(run_dir.glob("*_samples.csv"))
    if not csv_files:
        raise RuntimeError(f"No *_samples.csv found in {run_dir}")
    csv_path = csv_files[0]

    k90_vals = []
    k99_vals = []
    ripple_vals = []
    peak_vals = []
    steps_vals = []
    pass_vals = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column names from your screenshot
            k90_vals.append(float(row["k90_ms"]))
            k99_vals.append(float(row["k99_ms"]))
            ripple_vals.append(float(row["ripple_pct"]))  # still in percent, that’s fine
            peak_vals.append(float(row["peak"]))
            steps_vals.append(float(row["steps"]))

            # CSV has TRUE/FALSE strings
            pass_str = row["pass"].strip().upper()
            pass_vals.append(pass_str == "TRUE")

    metrics = {
        "k90_ms": mean(k90_vals),
        "k99_ms": mean(k99_vals),
        "ripple": mean(ripple_vals),  # this is ripple_pct average
        "peak": mean(peak_vals),
        "steps": mean(steps_vals),
        "pass": all(pass_vals),
    }
    return metrics


def main():
    if len(sys.argv) != 2:
        print("Usage: dnf_eval_wrapper.py CONFIG_YAML", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]

    # 1) Call the real DNF core (bb_run) with this config
    cmd = [
        sys.executable, "-m", "python_env.bb_run",
        "--config", config_path,
        "--cert", "configs/certification.yaml",
        "--datasets", "configs/datasets.yaml",
        "--dataset", "ycbsight_real",
        "--split", "test",
        "--out", str(RUNS_ROOT),
        "--max-frames", "80",
        "--per-sample",
        "--sample-steps", "200",
        "--print-every", "999999",   # basically mute
        # "--save-traces",           # enable if you really need traces
    ]

    # Remember current latest run dir mtime so we can pick the new one
    before_dirs = {p.name for p in RUNS_ROOT.iterdir()} if RUNS_ROOT.exists() else set()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[WRAPPER] bb_run failed", file=sys.stderr)
        sys.exit(result.returncode)

    # 2) Find the new run directory (the one not in before_dirs or the newest)
    time.sleep(0.5)  # tiny delay to ensure timestamps update
    candidates = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    new_dirs = [p for p in candidates if p.name not in before_dirs]

    if new_dirs:
        run_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
    else:
        # fallback: just use the newest one
        run_dir = find_latest_run_dir()

    # 3) Parse CSV and compute metrics
    metrics = read_metrics_from_csv(run_dir)

    # 4) Print JSON for C++
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
