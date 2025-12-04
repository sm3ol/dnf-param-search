# python_env/bb_core.py
# -*- coding: utf-8 -*-
"""
Black-box inner-loop runner for the 1-D DNF.

Public API:
    step_dnf_stream(Iext_stream, cfg, collect_full=False) -> dict
"""

from typing import Mapping, Iterable
import numpy as np
from core.dft_core import DynamicField


def _to_f32(x):
    return np.asarray(x, dtype=np.float32)


def step_dnf_stream(Iext_stream: Iterable[np.ndarray], cfg: Mapping, collect_full: bool = False) -> dict:
    """
    Run the DNF on a stream of external inputs.

    Parameters
    ----------
    Iext_stream : iterable of (N,) float32 arrays
    cfg : Mapping with DNF params (N, tau, dt, h, beta, theta, w_exc, sigma_exc, w_inh, sigma_inh, ...)
    collect_full : bool
        If True, return full space–time fields 'u' and 'a' and the input 'I'.

    Returns
    -------
    dict with keys:
        't','u_max','a_max','peak_idx','dt','tau'
        plus (if collect_full) 'u','a','I'
    """
    # --- Required params
    N      = int(cfg["N"])
    tau    = float(cfg["tau"])
    dt     = float(cfg["dt"])
    h      = float(cfg["h"])
    beta   = float(cfg["beta"])
    theta  = float(cfg["theta"])

    w_exc      = float(cfg["w_exc"])
    sigma_exc  = float(cfg["sigma_exc"])
    w_inh      = float(cfg["w_inh"])
    sigma_inh  = float(cfg["sigma_inh"])

    boundary    = str(cfg.get("boundary", "wrap"))
    kernel_norm = str(cfg.get("kernel_norm", "zero_mean_l2"))
    max_step    = float(cfg.get("max_step", 0.5))

    # --- Build field
    field = DynamicField(
        size=N, tau=tau, resting_level=h,
        boundary=boundary, beta=beta, threshold=theta,
        w_exc=w_exc, sigma_exc=sigma_exc,
        w_inh=w_inh, sigma_inh=sigma_inh,
        kernel_norm=kernel_norm, max_step=max_step,
    )

    # --- Buffers
    t_list: list[int] = []
    u_max:  list[float] = []
    a_max:  list[float] = []
    peak_idx: list[int] = []

    u_series = [] if collect_full else None
    a_series = [] if collect_full else None
    I_series = [] if collect_full else None

    # --- Main loop
    for step, I in enumerate(Iext_stream):
        I = _to_f32(I).reshape(-1)
        if I.shape[0] != N:
            raise ValueError(f"I_ext shape {I.shape} != (N,) = ({N},)")

        u = field.update(I, dt=dt)
        # activation with the same nonlinearity used inside the field
        a = field._sigmoid(u, beta=field.beta, threshold=field.threshold)

        t_list.append(step)
        u_max.append(float(u.max()))
        a_max.append(float(a.max()))
        peak_idx.append(int(np.argmax(u)))

        if collect_full:
            u_series.append(u.astype(np.float32, copy=True))
            a_series.append(a.astype(np.float32, copy=True))
            I_series.append(I.astype(np.float32, copy=True))

    out = {
        "t":        np.asarray(t_list, dtype=np.int32),
        "u_max":    np.asarray(u_max, dtype=np.float32),
        "a_max":    np.asarray(a_max, dtype=np.float32),
        "peak_idx": np.asarray(peak_idx, dtype=np.int32),
        "dt":       dt,
        "tau":      tau,
    }
    if collect_full:
        out["u"] = np.stack(u_series, axis=0) if u_series else np.zeros((0, N), np.float32)
        out["a"] = np.stack(a_series, axis=0) if a_series else np.zeros((0, N), np.float32)
        out["I"] = np.stack(I_series, axis=0) if I_series else np.zeros((0, N), np.float32)
    return out
