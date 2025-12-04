#!/usr/bin/env python3
import sys
import json
import math
import random


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    if len(sys.argv) != 2:
        print("Usage: toy_eval.py PARAMS_JSON", file=sys.stderr)
        sys.exit(1)

    params_path = sys.argv[1]
    with open(params_path, "r") as f:
        p = json.load(f)

    # Extract DNF parameters (names must match DNFParams::toJSON)
    tau = float(p["tau"])
    dt = float(p["dt"])
    beta = float(p["beta"])
    theta = float(p["theta"])
    h = float(p["h"])
    w_exc = float(p["w_exc"])
    sigma_exc = float(p["sigma_exc"])
    w_inh = float(p["w_inh"])
    sigma_inh = float(p["sigma_inh"])
    intensity = float(p["intensity_scale_factor"])
    gauss_width = int(p["gaussian_stimulus_width"])
    N = int(p["N"])

    # ------------------------------------------------------------------
    # Toy formulas that *behave like* the real system, but are cheap.
    # You can tweak these later if you want them closer to reality.
    # ------------------------------------------------------------------

    # 1) Latencies k90_ms, k99_ms (smaller when tau & dt are small, intensity higher)
    # Base scale ~ 60 ms when tau = 0.01, dt ≈ 0.001, intensity ≈ 7
    tau_ref = 0.010
    dt_ref = 0.001
    intensity_ref = 7.0

    tau_factor = tau / tau_ref          # >1 => slower
    dt_factor = dt / dt_ref             # >1 => coarser steps (slightly slower)
    intensity_factor = intensity_ref / max(intensity, 1e-6)  # larger intensity => faster

    base_k99 = 60.0 * tau_factor * (0.7 + 0.6 * dt_factor) * intensity_factor

    # Add a small noise to avoid ties
    noise_k99 = random.gauss(0.0, 3.0)
    k99_ms = clamp(base_k99 + noise_k99, 5.0, 400.0)

    # k90 is earlier than k99, roughly ~70% of it plus small noise
    noise_k90 = random.gauss(0.0, 2.0)
    k90_ms = clamp(0.7 * k99_ms + noise_k90, 1.0, k99_ms)

    # 2) Ripple [%]: higher if excitation dominates inhibition, lower if sigma_inh >> sigma_exc.
    exc_inh_ratio = w_exc / max(w_inh, 1e-3)
    sigma_gap = sigma_inh - sigma_exc  # larger gap = broader inhibition = less ripple

    base_ripple = 1.0 * (exc_inh_ratio - 1.0) - 0.08 * (sigma_gap - 4.0)
    # Slight dependence on beta/theta (steeper / lower threshold can increase ripple)
    base_ripple += 0.05 * (beta - 5.0) - 5.0 * (theta - 0.05)

    noise_ripple = random.gauss(0.0, 0.3)
    ripple = clamp(base_ripple + noise_ripple, 0.0, 10.0)  # percent

    # 3) Peak activation: grows with w_exc & intensity, drops if bias too negative or inhibition too strong
    # h is negative; less negative (closer to 0) usually gives higher peak.
    h_term = (h + 0.6) / 0.55  # h in [-0.60, -0.05] -> roughly [0, 1]
    peak_raw = 0.4 \
        + 0.10 * (w_exc - 1.5) \
        + 0.05 * (intensity - intensity_ref) \
        + 0.15 * h_term \
        - 0.07 * (w_inh - 1.0)

    noise_peak = random.gauss(0.0, 0.05)
    peak = clamp(peak_raw + noise_peak, 0.0, 1.5)

    # 4) Steps to settle: roughly k99 / dt, plus a mild dependency on N and gauss_width
    dt_ms = dt * 1000.0
    if dt_ms <= 0:
        steps = 1000.0
    else:
        steps = k99_ms / dt_ms

    # Adjust steps by spatial size and probe width (bigger maps need more iterations)
    steps *= (0.8 + 0.4 * ((N / 100.0)))
    steps *= (0.9 + 0.1 * (gauss_width / 7.0))

    noise_steps = random.gauss(0.0, 10.0)
    steps = clamp(steps + noise_steps, 10.0, 5000.0)

    # ------------------------------------------------------------------
    # Certification gates from the problem statement:
    #   - k99_ms <= 100 ms (speed gate)
    #   - peak   >= 0.60   (SNR gate)
    #   - ripple <= 2%     (stability gate)
    # ------------------------------------------------------------------
    pass_gate = (k99_ms <= 100.0) and (peak >= 0.60) and (ripple <= 2.0)

    metrics = {
        "k90_ms": float(k90_ms),
        "k99_ms": float(k99_ms),
        "ripple": float(ripple),
        "peak": float(peak),
        "steps": float(steps),
        "pass": bool(pass_gate),
    }

    # Print as JSON to stdout for the C++ side to parse
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
