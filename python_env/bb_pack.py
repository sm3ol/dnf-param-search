# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 11:00:32 2025

@author: iT
"""

# python_env/bb_pack.py
# -*- coding: utf-8 -*-
"""
Per-sample pack writer: arrays + figures + metrics + manifest.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python_env.bb_metrics import (
    steady_value, kx_time, settle_time, fwhm_bins
)
from python_env.bb_config import config_hash


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_sample_pack(
    root_dir: Path,
    dataset: str,
    seq_id: str,
    sample_idx: int,
    sim: dict,
    I_1d: np.ndarray,
    cfg_obj,
    gates: dict | None = None,
) -> dict:
    """
    Write:
      - fields.npz  (t_ms, U, A, stim, meta)
      - stimulus.png, rise_curve.png, a_heatmap.png, u_heatmap.png, bump_profile.png
      - metrics.json
      - manifest.json

    Returns the metrics dict (also written).
    """
    gates = gates or {}
    N = int(sim["U"].shape[1])
    T = int(sim["U"].shape[0])
    dt_ms = float(sim["dt"] * 1000.0)
    t_ms = np.arange(T, dtype=np.float32) * dt_ms

    U = sim["U"]             # [T,N]
    A = sim["A"] if "A" in sim else 1.0/(1.0+np.exp(-cfg_obj.beta*(U-cfg_obj.theta)))
    stim = np.asarray(I_1d, dtype=np.float32)

    # ---- metrics (steady window = max(20%T, 1 ms))
    tail_len = max(int(0.2*T), int(round(1.0 / max(dt_ms, 1e-6))))
    tail_len = min(max(tail_len, 1), T)
    a_max = A.max(axis=1)
    u_max = U.max(axis=1)
    a_ss = steady_value(a_max, tail_frac=0.2, min_len=tail_len)
    k10 = kx_time(a_max, t_ms, a_ss, 10)
    k50 = kx_time(a_max, t_ms, a_ss, 50)
    k90 = kx_time(a_max, t_ms, a_ss, 90)
    t_settle = settle_time(a_max, t_ms, a_ss, eps=0.01, dwell_ms=1.0)

    a_tail = a_max[-tail_len:]
    overshoot = 100.0 * (a_tail.max() - a_ss) / max(a_ss, 1e-12)
    ripple = 100.0 * (a_tail.max() - a_tail.min()) / max(a_ss, 1e-12)

    a_ss_prof = A[-tail_len:, :].mean(axis=0)
    peak_idx = int(np.argmax(a_ss_prof))
    width = fwhm_bins(a_ss_prof)

    x = np.arange(N, dtype=np.float32)
    centroid = float(np.average(x, weights=a_ss_prof))
    traj_centroid = np.average(x[None, :], weights=A[-tail_len:, :], axis=1)
    jitter = float(np.std(traj_centroid))

    # simple status rules (can be adjusted or replaced by cfg-based certification)
    reasons = []
    status = "PASS"
    if k90 is None: status, reasons = "UNSTABLE", ["no_k90"]
    else:
        if k90 > gates.get("k90_ms", 5.0): reasons.append("k90>gate")
        if overshoot > 10.0: reasons.append("overshoot>10%")
        if ripple > 2.0: reasons.append("ripple>2%")
        if (width is None) or not (1 <= width <= 20): reasons.append("FWHM_out_of_range")
        if reasons: status = "FAIL"

    metrics = dict(
        dataset=dataset, seq_id=seq_id, sample_idx=sample_idx,
        N=N, dt_ms=dt_ms, T_ms=float(t_ms[-1] - t_ms[0]) if T > 1 else 0.0,
        a_max_ss=float(a_ss), k10_ms=k10, k50_ms=k50, k90_ms=k90, t_settle_ms=t_settle,
        overshoot_pct=float(overshoot), ripple_pct=float(ripple),
        fwhm_px=None if width is None else float(width),
        centroid_px=float(centroid), centroid_jitter_px=float(jitter),
        peak_index=peak_idx,
        status=status, reasons=reasons,
        seed=getattr(cfg_obj, "seed", None),
        config_hash=config_hash(cfg_obj),
    )

    # ---- paths
    sample_dir = _ensure_dir(root_dir / "samples" / seq_id / f"sample_{sample_idx:04d}")
    sid = f"sample_{sample_idx:04d}"

    # ---- arrays
    np.savez_compressed(
        sample_dir / f"{sid}_fields.npz",
        t_ms=t_ms.astype(np.float32),
        u=U.astype(np.float32),
        a=A.astype(np.float32),
        stim=stim.astype(np.float32),
        meta=dict(N=N, dt_ms=dt_ms, config_hash=metrics["config_hash"], sample_id=sid)
    )

    # ---- figures
    # stimulus
    plt.figure(figsize=(6,3)); plt.plot(stim); plt.title("Stimulus (N=%d)"%N)
    plt.xlabel("Position"); plt.ylabel("Intensity"); plt.tight_layout()
    plt.savefig(sample_dir / f"{sid}_stimulus.png", dpi=150); plt.close()

    # rise curve
    plt.figure(figsize=(6,3))
    plt.plot(t_ms, a_max, label="a_max"); plt.plot(t_ms, u_max, label="u_max")
    for val, lab in [(k10,"k10"), (k50,"k50"), (k90,"k90")]:
        if val is not None: plt.axvline(val, ls="--", alpha=0.6, label=lab)
    plt.axhline(a_ss, ls=":", alpha=0.6, label="steady")
    ttl = f"Activation Rise  |  k90={None if k90 is None else f'{k90:.2f}'} ms"
    plt.title(ttl); plt.xlabel("ms"); plt.ylabel("peak (a.u.)"); plt.legend()
    plt.tight_layout(); plt.savefig(sample_dir / f"{sid}_rise_curve.png", dpi=150); plt.close()

    # heatmaps
    extent = [0, N, float(t_ms[0]), float(t_ms[-1] if T>1 else 0.0)]
    plt.figure(figsize=(9,3)); plt.imshow(A, aspect="auto", origin="lower",
        extent=extent); plt.colorbar(label="a"); plt.xlabel("Position")
    plt.ylabel("Time (ms)"); plt.title("Activation a(x,t)"); plt.tight_layout()
    plt.savefig(sample_dir / f"{sid}_a_heatmap.png", dpi=150); plt.close()

    plt.figure(figsize=(9,3)); plt.imshow(U, aspect="auto", origin="lower",
        extent=extent); plt.colorbar(label="u"); plt.xlabel("Position")
    plt.ylabel("Time (ms)"); plt.title("DNF field u(x,t)"); plt.tight_layout()
    plt.savefig(sample_dir / f"{sid}_u_heatmap.png", dpi=150); plt.close()

    # steady bump profile + FWHM
    plt.figure(figsize=(6,3)); plt.plot(a_ss_prof); plt.title("Steady bump (a)")
    plt.xlabel("Position"); plt.ylabel("a")
    if width is not None:
        half = 0.5 * float(a_ss_prof.max())
        plt.axhline(half, ls=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(sample_dir / f"{sid}_bump_profile.png", dpi=150); plt.close()

    # ---- metrics.json + manifest.json
    (sample_dir / f"{sid}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "sample_id": sid,
        "artifacts": [
            f"{sid}_fields.npz",
            f"{sid}_stimulus.png",
            f"{sid}_rise_curve.png",
            f"{sid}_a_heatmap.png",
            f"{sid}_u_heatmap.png",
            f"{sid}_bump_profile.png",
            f"{sid}_metrics.json",
        ]
    }
    (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics
