import json
import os
from pathlib import Path

import optuna
from optuna.samplers import MOTPESampler

from toy_eval import evaluate_params


# ---------------------------------------------------------------------
# Search space: must match DNFParams ranges + constraints
# ---------------------------------------------------------------------

def sample_dnf_params(trial: optuna.trial.Trial) -> dict:
    # Continuous ranges (same as DNFParams::isValid)
    tau = trial.suggest_float("tau", 0.005, 0.020)
    dt = trial.suggest_float("dt", 0.0005, 0.0020)
    beta = trial.suggest_float("beta", 2.0, 7.0)
    theta = trial.suggest_float("theta", 0.0, 0.1)
    h = trial.suggest_float("h", -0.60, -0.05)
    w_exc = trial.suggest_float("w_exc", 0.8, 3.0)
    sigma_exc = trial.suggest_float("sigma_exc", 2.0, 6.0)
    w_inh = trial.suggest_float("w_inh", 0.3, 2.0)
    intensity_scale_factor = trial.suggest_float("intensity_scale_factor", 3.0, 12.0)

    # sigma_inh > sigma_exc + 1, within [3, 20]
    sigma_inh_min = max(3.0, sigma_exc + 1.0000001)
    sigma_inh = trial.suggest_float("sigma_inh", sigma_inh_min, 20.0)

    # Discrete
    gaussian_stimulus_width = trial.suggest_int("gaussian_stimulus_width", 3, 11)
    N = trial.suggest_categorical("N", [50, 100, 150, 200])

    # Enforce alpha = dt / tau in (0, 0.2)
    alpha = dt / tau
    if not (0.0 < alpha < 0.2):
        # Tell Optuna this sample is invalid; it will resample efficiently.
        raise optuna.TrialPruned(f"alpha={alpha:.4f} out of (0,0.2)")

    return {
        "tau": tau,
        "dt": dt,
        "beta": beta,
        "theta": theta,
        "h": h,
        "w_exc": w_exc,
        "sigma_exc": sigma_exc,
        "w_inh": w_inh,
        "sigma_inh": sigma_inh,
        "intensity_scale_factor": intensity_scale_factor,
        "gaussian_stimulus_width": int(gaussian_stimulus_width),
        "N": int(N),
    }


# ---------------------------------------------------------------------
# Multi-objective: same 4 objectives as NSGA-II
# ---------------------------------------------------------------------

def objectives_from_metrics(metrics: dict):
    # All MINIMIZED
    k90 = metrics["k90_ms"]
    ripple = metrics["ripple"]
    steps = metrics["steps"]
    overshoot = max(0.0, metrics["peak"] - 1.0)

    BIG = 1e6
    if not metrics["pass"]:
        # heavily penalize spec failures
        return [
            k90 + BIG,
            ripple + BIG,
            steps + BIG,
            overshoot + BIG,
        ]
    return [k90, ripple, steps, overshoot]


def motpe_objective(trial: optuna.trial.Trial):
    # Sample valid DNF params
    params = sample_dnf_params(trial)

    # Evaluate via toy_eval core
    metrics = evaluate_params(params)

    # Turn into 4-dimensional objective vector
    return objectives_from_metrics(metrics)


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main():
    out_dir = Path("artifacts") / "pareto"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "motpe_pareto.jsonl"

    sampler = MOTPESampler(seed=42)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    n_trials = 100  # you can bump this later
    print(f"Running MOTPE with {n_trials} trials...")
    study.optimize(motpe_objective, n_trials=n_trials, show_progress_bar=True)

    # Extract Pareto front (multi-objective best trials)
    pareto_trials = study.best_trials

    print(f"\nFound {len(pareto_trials)} Pareto-optimal trials.")
    print(f"Saving them to {out_path}")

    with out_path.open("w") as f:
        for t in pareto_trials:
            row = {
                "trial_id": t.number,
                "values": t.values,
                "params": t.params,
            }
            f.write(json.dumps(row) + "\n")

    # Also print a nice summary of the "best" (smallest sum of objectives)
    best_trial = min(pareto_trials, key=lambda tr: sum(tr.values))
    print("\n=== MOTPE representative best ===")
    print("Objectives [k90, ripple, steps, overshoot]:", best_trial.values)
    print("Params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
