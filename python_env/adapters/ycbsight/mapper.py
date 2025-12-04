# python_env/adapters/ycbsight/mapper.py
# -*- coding: utf-8 -*-
"""
2D → 1D mapping utilities for YCBSight frames.

Core function:
    frame_to_Iext_1d(frame, N, mode="centroid", sigma_bins=3.0,
                     intensity_scale_mode="peak", intensity_gain=1.0,
                     intensity_floor=0.0, intensity_ceiling=1.0,
                     input_noise_std=0.0)

Steps:
1) Pick a representative position in the 2D frame (centroid or argmax).
2) Map the x-position in [0..W-1] to a 1-D field index in [0..N-1].
3) Place a Gaussian bump (σ in bins) centered at that index.
4) Set amplitude based on frame intensity with simple, robust modes.
"""

from __future__ import annotations
from typing import Literal, Tuple
import numpy as np


Mode = Literal["centroid", "argmax"]
ScaleMode = Literal["fixed", "peak", "mean", "p95", "sum"]


def _gaussian_1d(center_idx: float, N: int, sigma_bins: float) -> np.ndarray:
    x = np.arange(N, dtype=np.float32)
    s = max(1e-6, float(sigma_bins))
    return np.exp(-0.5 * ((x - center_idx) / s) ** 2, dtype=np.float32)


def _choose_position(frame: np.ndarray, mode: Mode) -> Tuple[float, float]:
    """Return (x_idx, intensity_proxy in [0,1])."""
    if mode == "argmax":
        idx = int(np.argmax(frame))
        H, W = frame.shape
        y, x = divmod(idx, W)
        inten = float(frame[y, x])
        return float(x), float(inten)
    # centroid (default)
    w = frame.astype(np.float64, copy=False)
    total = w.sum()
    if total <= 0:
        H, W = frame.shape
        return float((W - 1) / 2.0), 0.0
    H, W = frame.shape
    xs = np.arange(W, dtype=np.float64)
    # marginal along x
    mx = (w.sum(axis=0) + 1e-12)
    x = float((xs * mx).sum() / mx.sum())
    # intensity proxy near the centroid (robust: local 5×5 window)
    cx, cy = int(round(x)), int(round((H - 1) / 2.0))
    x0, x1 = max(0, cx - 2), min(W, cx + 3)
    y0, y1 = max(0, cy - 2), min(H, cy + 3)
    inten = float(np.clip(w[y0:y1, x0:x1].mean() if (y1 > y0 and x1 > x0) else 0.0, 0.0, 1.0))
    return x, inten


def _amplitude_from_frame(frame: np.ndarray, mode: ScaleMode, gain: float,
                          floor: float, ceil: float) -> float:
    if mode == "fixed":
        amp = 1.0
    elif mode == "peak":
        amp = float(frame.max())
    elif mode == "mean":
        amp = float(frame.mean())
    elif mode == "p95":
        amp = float(np.percentile(frame, 95.0))
    elif mode == "sum":
        amp = float(frame.sum()) / float(max(1, frame.size))
    else:
        raise ValueError(f"Unknown intensity_scale_mode '{mode}'")
    amp = np.clip(amp * float(gain), float(floor), float(ceil))
    return float(amp)


def frame_to_Iext_1d(
    frame: np.ndarray,
    N: int,
    mode: Mode = "centroid",
    sigma_bins: float = 3.0,
    intensity_scale_mode: ScaleMode = "peak",
    intensity_gain: float = 1.0,
    intensity_floor: float = 0.0,
    intensity_ceiling: float = 1.0,
    input_noise_std: float = 0.0,
) -> np.ndarray:
    """
    Map an H×W frame (float32 in [0,1]) to a length-N I_ext (float32).

    Parameters
    ----------
    mode : "centroid" or "argmax"
        Choose how to pick the representative x-position.
    sigma_bins : float
        Gaussian width on the 1-D field, in bins.
    intensity_scale_mode : "fixed"|"peak"|"mean"|"p95"|"sum"
        How to set the amplitude from the frame.
    intensity_gain : float
        Multiplier on the amplitude after scaling mode.
    intensity_floor, intensity_ceiling : float
        Clamp amplitude into [floor, ceiling].
    input_noise_std : float
        Additive Gaussian noise std (a.u.) on the final I (for stress tests).

    Returns
    -------
    I : np.ndarray, shape (N,), dtype float32
    """
    if frame.ndim != 2:
        raise ValueError(f"frame_to_Iext_1d expects 2D array, got shape {frame.shape}")
    H, W = frame.shape
    x_pos, local_inten = _choose_position(frame, mode=mode)
    # map from [0..W-1] -> [0..N-1]
    center_idx = (x_pos / max(1.0, (W - 1))) * (N - 1)
    gauss = _gaussian_1d(center_idx=center_idx, N=int(N), sigma_bins=float(sigma_bins)).astype("float32")

    amp = _amplitude_from_frame(
        frame, mode=intensity_scale_mode, gain=float(intensity_gain),
        floor=float(intensity_floor), ceil=float(intensity_ceiling)
    )
    I = amp * gauss
    if input_noise_std and input_noise_std > 0.0:
        I = I + np.random.normal(0.0, float(input_noise_std), size=I.shape).astype(I.dtype)
    return I.astype("float32", copy=False)
