# python_env/bb_metrics.py
# -*- coding: utf-8 -*-
"""
Metrics & certification helpers for the DNF black box (YAML-driven).

Primary entry point:
    certify_sequence(sim: dict, cfg_obj, gates_from_yaml: dict) -> dict

Behavior
--------
- A check is ENABLED iff:
    * gates['use'][name] is True, OR (if 'use' is absent) the required
      gate key(s) for that check are present in the YAML.
- If a gate is disabled / not present, that check is skipped and does not
  affect pass/fail.
- Percent-like gates (e.g., max_ripple) accept either fractions (0..1)
  or percents (0..100); they are normalized to percent for comparison
  against ripple_pct (%).
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np


# -------------------------- small helpers --------------------------

def _steady_value(a_max: np.ndarray, t_ms: np.ndarray,
                  frac: float = 0.2, min_ms: float = 1.0) -> Tuple[float, int, int]:
    """Median over the last window as steady-state; returns (steady, start_idx, tail_len)."""
    if a_max.size == 0:
        return 0.0, 0, 0
    dt = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else (t_ms[0] if t_ms.size else 1.0)
    tail = max(int(len(t_ms) * frac), int(min_ms / max(dt, 1e-9)))
    tail = max(1, min(tail, len(t_ms)))
    start = len(t_ms) - tail
    aa = a_max[-tail:]
    return float(np.median(aa)), start, tail


def _first_cross_time(t_ms: np.ndarray, y: np.ndarray, thr: float) -> Optional[float]:
    if y.size == 0:
        return None
    mask = (y >= thr)
    if not np.any(mask):
        return None
    idx = int(np.argmax(mask))
    return float(t_ms[idx])


def _pct_from_gate_value(val: float) -> float:
    """Allow gates to be specified as fraction (0..1) or percent (0..100)."""
    if val is None:
        return None
    return float(val) * 100.0 if float(val) <= 1.0 else float(val)


# -------- circular-safe spatial helpers (handle ring seam properly) --------

def _fwhm(profile: np.ndarray) -> Optional[float]:
    """FWHM on a circular 1-D field (rotate so the global peak is contiguous)."""
    if profile is None or profile.size == 0:
        return None
    peak = float(profile.max())
    if peak <= 0:
        return None
    thr = 0.5 * peak
    pk = int(np.argmax(profile))
    m = np.roll(profile >= thr, -pk)  # rotate so peak is at index 0
    if not m.any():
        return 0.0
    if m.all():
        return float(m.size)
    # length of the leading True run
    # handle case where index 0 might still be False after rotation (quantization)
    if not m[0]:
        i0 = int(np.argmax(m))  # first True
        m = np.roll(m, -i0)
    length = int(np.argmax(~m))
    if length == 0 and m.all():
        length = m.size
    return float(length)


def _count_islands(profile: np.ndarray) -> Optional[int]:
    """Number of contiguous ≥50%-of-peak regions on a circular field."""
    if profile is None or profile.size == 0:
        return None
    peak = float(profile.max())
    if peak <= 0:
        return 0
    thr = 0.5 * peak
    m = (profile >= thr)
    if not m.any():
        return 0
    if m.all():
        return 1
    rising = (~np.roll(m, 1)) & m  # circular rising edges
    return int(rising.sum())


# ----------------------- main certification -----------------------

def certify_sequence(sim: Dict[str, Any], cfg_obj, gates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce metrics (time-domain + spatial) and a pass/fail flag using ONLY
    the gates present in `gates` (YAML). Missing gates -> check disabled.
    """
    # ---- timebase
    t_idx = sim["t"].astype(np.int32)
    dt = float(sim["dt"])
    t_ms = t_idx * dt * 1000.0

    # ---- traces
    a_max = np.asarray(sim["a_max"], dtype=np.float32)
    u_max = np.asarray(sim.get("u_max", np.zeros_like(a_max)), dtype=np.float32)

    # measurement options (optional in YAML)
    steady_cfg = gates.get("steady_window", {})
    steady_frac = float(steady_cfg.get("frac", 0.2))
    steady_min_ms = float(steady_cfg.get("min_ms", 1.0))

    # compute steady & tail indices once
    a_ss, steady_start, tail = _steady_value(a_max, t_ms, frac=steady_frac, min_ms=steady_min_ms)

    # base metrics
    k90_ms = _first_cross_time(t_ms, a_max, 0.90 * a_ss) if a_ss > 0 else None
    k99_ms = _first_cross_time(t_ms, a_max, 0.99 * a_ss) if a_ss > 0 else None

    if len(t_ms) > 1 and tail > 0:
        tail_seg = a_max[-tail:]
        ripple_pct = 0.0 if a_ss <= 0 else float((tail_seg.max() - tail_seg.min()) / max(a_ss, 1e-12) * 100.0)
    else:
        ripple_pct = 0.0

    # spatial (only if 'a' present)
    a = sim.get("a", None)
    if isinstance(a, np.ndarray) and a.size:
        a_ss_profile = a[-tail:, :].mean(axis=0) if tail > 0 else a[-1, :]
        fwhm_bins = _fwhm(a_ss_profile)
        islands = _count_islands(a_ss_profile)
    else:
        a_ss_profile = None
        fwhm_bins = None
        islands = None

    # optional extras (computed only if their gates are requested)
    overshoot_pct = None
    if ("overshoot_pct" in gates) or (gates.get("use", {}).get("overshoot_pct", False)):
        if a_ss > 0:
            overshoot_pct = float((float(a_max.max()) - a_ss) / a_ss * 100.0)
        else:
            overshoot_pct = 0.0

    centroid_jitter_px = None
    if ("centroid_jitter_px" in gates) or (gates.get("use", {}).get("centroid_jitter_px", False)):
        if isinstance(a, np.ndarray) and a.size and tail > 0:
            x = np.arange(a.shape[1], dtype=np.float32)
            W = a[-tail:, :]
            num = (W * x[None, :]).sum(axis=1)
            den = (W.sum(axis=1) + 1e-12)
            traj_cent = num / den
            centroid_jitter_px = float(np.std(traj_cent))
        else:
            centroid_jitter_px = 0.0

    # assemble metric payload
    m = {
        "steps": int(len(t_idx)),
        "k90_ms": float(k90_ms) if k90_ms is not None else float("inf"),
        "k99_ms": float(k99_ms) if k99_ms is not None else float("inf"),
        "ripple_pct": float(ripple_pct),
        "peak": float(a_ss),
        "fwhm_bins": None if fwhm_bins is None else float(fwhm_bins),
        "islands": islands if islands is not None else None,
    }
    if overshoot_pct is not None:
        m["overshoot_pct"] = overshoot_pct
    if centroid_jitter_px is not None:
        m["centroid_jitter_px"] = centroid_jitter_px

    # ------------- resolve which checks to use (YAML only) -------------
    use = gates.get("use", {})  # optional
    def enabled(name: str, *required_keys) -> bool:
        if name in use:
            return bool(use[name])
        # if no 'use' section: enable only if all required gate keys are present
        return all(k in gates for k in required_keys)

    # thresholds gathered from YAML (no defaults)
    G: Dict[str, Any] = {}
    checks: Dict[str, bool] = {}
    reasons = []

    # latency
    if enabled("k90_ms", "k90_ms"):
        G["k90_ms"] = float(gates["k90_ms"])
        ok = (m["k90_ms"] <= G["k90_ms"])
        checks["k90_pass"] = ok
        if not ok:
            reasons.append(f"k90 {m['k90_ms']:.2f} > {G['k90_ms']}")
    if enabled("k99_ms", "k99_ms"):
        G["k99_ms"] = float(gates["k99_ms"])
        ok = (m["k99_ms"] <= G["k99_ms"])
        checks["k99_pass"] = ok
        if not ok:
            reasons.append(f"k99 {m['k99_ms']:.2f} > {G['k99_ms']}")

    # stability / selectivity
    if enabled("ripple_pct", "max_ripple"):
        G["max_ripple"] = _pct_from_gate_value(float(gates["max_ripple"]))
        ok = (m["ripple_pct"] <= G["max_ripple"])
        checks["ripple_pass"] = ok
        if not ok:
            reasons.append(f"ripple {m['ripple_pct']:.2f}% > {G['max_ripple']:.2f}%")
    if enabled("min_peak", "min_peak"):
        G["min_peak"] = float(gates["min_peak"])
        ok = (m["peak"] >= G["min_peak"])
        checks["peak_pass"] = ok
        if not ok:
            reasons.append(f"peak {m['peak']:.3f} < {G['min_peak']:.3f}")

    if "overshoot_pct" in m and enabled("overshoot_pct", "overshoot_pct"):
        G["overshoot_pct"] = float(gates["overshoot_pct"])
        ok = (m["overshoot_pct"] <= G["overshoot_pct"])
        checks["overshoot_pass"] = ok
        if not ok:
            reasons.append(f"overshoot {m['overshoot_pct']:.2f}% > {G['overshoot_pct']:.2f}%")

    if "centroid_jitter_px" in m and enabled("centroid_jitter_px", "centroid_jitter_px"):
        G["centroid_jitter_px"] = float(gates["centroid_jitter_px"])
        ok = (m["centroid_jitter_px"] <= G["centroid_jitter_px"])
        checks["centroid_jitter_pass"] = ok
        if not ok:
            reasons.append(f"centroid_jitter {m['centroid_jitter_px']:.2f} > {G['centroid_jitter_px']:.2f}")

    # spatial (width / islands)
    # Support either fwhm_bins: [min,max] or min_fwhm_bins / max_fwhm_bins
    if enabled("fwhm_bins", "fwhm_bins") or enabled("fwhm_bins", "min_fwhm_bins", "max_fwhm_bins"):
        if "fwhm_bins" in gates and isinstance(gates["fwhm_bins"], (list, tuple)) and len(gates["fwhm_bins"]) == 2:
            G["min_fwhm_bins"], G["max_fwhm_bins"] = int(gates["fwhm_bins"][0]), int(gates["fwhm_bins"][1])
        else:
            if "min_fwhm_bins" in gates: G["min_fwhm_bins"] = int(gates["min_fwhm_bins"])
            if "max_fwhm_bins" in gates: G["max_fwhm_bins"] = int(gates["max_fwhm_bins"])
        if m["fwhm_bins"] is not None and "min_fwhm_bins" in G and "max_fwhm_bins" in G:
            ok = (G["min_fwhm_bins"] <= m["fwhm_bins"] <= G["max_fwhm_bins"])
            checks["fwhm_pass"] = ok
            if not ok:
                reasons.append(f"fwhm {int(m['fwhm_bins'])} ∉ [{G['min_fwhm_bins']},{G['max_fwhm_bins']}]")

    if enabled("islands", "max_islands"):
        G["max_islands"] = int(gates["max_islands"])
        if m["islands"] is not None:
            ok = (m["islands"] <= G["max_islands"])
            checks["islands_pass"] = ok
            if not ok:
                reasons.append(f"islands {int(m['islands'])} > {G['max_islands']}")

    # final status
    m.update(checks)
    m["pass"] = all(checks.values()) if checks else True
    m["reasons"] = "; ".join(reasons)
    m["gates_used"] = G
    return m
# # python_env/bb_metrics.py
# # -*- coding: utf-8 -*-
# """
# Metrics & certification helpers for the DNF black box.

# Primary entry point:
#     certify_sequence(sim: dict, cfg_obj, gates: dict) -> dict

# `sim` is the dictionary returned by `bb_core.step_dnf_stream(..)`.
# This module is safe to use for both sequence mode (no full fields)
# and per-sample mode (with full fields). When spatial fields are not
# available, spatial metrics (FWHM/islands) are set to None and do not
# affect pass/fail.

# Gate names (all optional; reasonable defaults applied):
#     k90_ms: float       (default: 5.0)
#     k99_ms: float       (default: 8.0)
#     max_ripple: float   (fraction or percent; default: 0.02 == 2%)
#     min_peak: float     (default: 0.05)
#     max_islands: int    (default: 1)
#     min_fwhm_bins: int  (default: 1)
#     max_fwhm_bins: int  (default: 1_000_000)
# """

# from __future__ import annotations
# from typing import Iterable, Optional, Dict, Any
# import numpy as np


# # -------------------------- small helpers --------------------------

# def _steady_value(a_max: np.ndarray, t_ms: np.ndarray, frac: float = 0.2, min_ms: float = 1.0):
#     """Median over the last window as steady-state."""
#     if a_max.size == 0:
#         return 0.0, 0
#     dt = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else (t_ms[0] if t_ms.size else 1.0)
#     tail = max(int(len(t_ms) * frac), int(min_ms / max(dt, 1e-9)))
#     tail = max(1, min(tail, len(t_ms)))
#     aa = a_max[-tail:]
#     return float(np.median(aa)), a_max.size - tail


# def _first_cross_time(t_ms: np.ndarray, y: np.ndarray, thr: float) -> Optional[float]:
#     if y.size == 0:
#         return None
#     idx = np.argmax(y >= thr)  # 0 if all False
#     return None if not np.any(y >= thr) else float(t_ms[idx])


# # def _fwhm(profile: np.ndarray) -> Optional[float]:
# #     if profile is None or profile.size == 0:
# #         return None
# #     peak = float(profile.max())
# #     if peak <= 0:
# #         return None
# #     half = 0.5 * peak
# #     above = np.where(profile >= half)[0]
# #     return None if above.size == 0 else float(above[-1] - above[0] + 1)


# # def _count_islands(profile: np.ndarray) -> Optional[int]:
# #     """Number of contiguous bumps above 50% of peak."""
# #     if profile is None or profile.size == 0:
# #         return None
# #     peak = float(profile.max())
# #     if peak <= 0:
# #         return 0
# #     mask = profile >= (0.5 * peak)
# #     # transitions from False->True count an island start
# #     starts = np.count_nonzero(np.logical_and(mask, np.logical_not(np.roll(mask, 1))))
# #     # if the first element is True, np.roll shifts a True to end; fix that:
# #     if mask[0]:
# #         starts += 0  # already counted correctly by the logical expression
# #     # if everything is False, result is 0
# #     # special case: all True -> one island
# #     return int(starts) if np.any(mask) else 0

# def _fwhm(profile: np.ndarray) -> Optional[float]:
#     """FWHM on a circular 1-D field (rotate so peak is centered)."""
#     if profile is None or profile.size == 0:
#         return None
#     peak = float(profile.max())
#     if peak <= 0:
#         return None
#     half = 0.5 * peak
#     pk = int(np.argmax(profile))
#     m = profile >= half
#     # rotate so the peak sits at index 0; avoids seam splitting
#     m = np.roll(m, -pk)
#     # count contiguous True run that starts at 0
#     # if there is any False at the beginning, find the first True segment
#     if not m.any():
#         return 0.0
#     # the run containing the peak is at the start after rotation
#     # length until the first False
#     length = int(np.argmax(~m)) if not (~m[0]) else 0
#     if length == 0:
#         # either m[0] is False (shouldn't happen) or all True
#         if m.all():
#             length = m.size
#         else:
#             # find first True index and then measure that run
#             i0 = int(np.argmax(m))
#             m2 = np.roll(m, -i0)
#             length = int(np.argmax(~m2))
#             if length == 0 and m2.all():
#                 length = m2.size
#     return float(length)


# def _count_islands(profile: np.ndarray) -> Optional[int]:
#     """Number of contiguous ≥50%-of-peak regions on a circular field."""
#     if profile is None or profile.size == 0:
#         return None
#     peak = float(profile.max())
#     if peak <= 0:
#         return 0
#     thr = 0.5 * peak
#     m = profile >= thr
#     if not m.any():
#         return 0
#     if m.all():
#         return 1
#     # circular rising edges
#     rising = (~np.roll(m, 1)) & m
#     return int(rising.sum())

# def _pct_from_gate_value(val: float) -> float:
#     """Allow gates to be specified as fraction (0..1) or percent (0..100)."""
#     if val is None:
#         return None
#     return val * 100.0 if val <= 1.0 else val


# # ----------------------- main certification -----------------------

# def certify_sequence(sim: Dict[str, Any], cfg_obj, gates: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Produce a compact set of metrics (time-domain + spatial) and a pass/fail flag.
#     Works whether or not sim contains full fields 'a'/'u'.
#     """
#     # Timebase
#     t_idx = sim["t"].astype(np.int32)
#     dt = float(sim["dt"])
#     t_ms = t_idx * dt * 1000.0

#     # Peak traces
#     a_max = np.asarray(sim["a_max"], dtype=np.float32)
#     u_max = np.asarray(sim["u_max"], dtype=np.float32)

#     # Steady value & rise times
#     a_ss, _ = _steady_value(a_max, t_ms)
#     k90_ms = _first_cross_time(t_ms, a_max, 0.90 * a_ss) if a_ss > 0 else None
#     k99_ms = _first_cross_time(t_ms, a_max, 0.99 * a_ss) if a_ss > 0 else None

#     # Ripple on steady window
#     if len(t_ms) > 1:
#         dt_ms = float(t_ms[1] - t_ms[0])
#         tail = max(int(len(t_ms) * 0.2), int(1.0 / max(dt_ms, 1e-9)))
#         tail = max(1, min(tail, len(t_ms)))
#         tail_seg = a_max[-tail:]
#         ripple_pct = 0.0 if a_ss <= 0 else float((tail_seg.max() - tail_seg.min()) / a_ss * 100.0)
#     else:
#         ripple_pct = 0.0

#     # Spatial profile (if available)
#     a = sim.get("a", None)
#     if isinstance(a, np.ndarray) and a.size:
#         a_ss_profile = a[-tail:, :].mean(axis=0) if len(t_ms) > 1 else a[-1, :]
#         fwhm_bins = _fwhm(a_ss_profile)
#         islands = _count_islands(a_ss_profile)
#     else:
#         a_ss_profile = None
#         fwhm_bins = None
#         islands = None

#     # Compose metrics dict
#     m = {
#         "steps": int(len(t_idx)),
#         "k90_ms": float(k90_ms) if k90_ms is not None else float("inf"),
#         "k99_ms": float(k99_ms) if k99_ms is not None else float("inf"),
#         "ripple_pct": float(ripple_pct),
#         "peak": float(a_ss),
#         "fwhm_bins": None if fwhm_bins is None else float(fwhm_bins),
#         "islands": islands if islands is not None else None,
#     }

#     # ------------- certification gates -------------
#     g = dict(
#         k90_ms=float(gates.get("k90_ms", 5.0)),
#         k99_ms=float(gates.get("k99_ms", 8.0)),
#         max_ripple=_pct_from_gate_value(float(gates.get("max_ripple", 0.02))),  # in %
#         min_peak=float(gates.get("min_peak", 0.05)),
#         max_islands=int(gates.get("max_islands", 1)),
#         min_fwhm_bins=int(gates.get("min_fwhm_bins", 1)),
#         max_fwhm_bins=int(gates.get("max_fwhm_bins", 1_000_000)),
#     )

#     checks = {
#         "k90_pass": (m["k90_ms"] <= g["k90_ms"]),
#         "k99_pass": (m["k99_ms"] <= g["k99_ms"]),
#         "ripple_pass": (m["ripple_pct"] <= g["max_ripple"]),
#         "peak_pass": (m["peak"] >= g["min_peak"]),
#     }

#     # Spatial checks only if available
#     if m["fwhm_bins"] is not None:
#         checks["fwhm_pass"] = (g["min_fwhm_bins"] <= m["fwhm_bins"] <= g["max_fwhm_bins"])
#     if m["islands"] is not None:
#         checks["islands_pass"] = (m["islands"] <= g["max_islands"])

#     m.update(checks)
#     m["pass"] = all(bool(v) for v in checks.values())

#     return m
