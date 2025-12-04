# core/dft_core.py
# -*- coding: utf-8 -*-
"""
Minimal, numerically stable 1-D Dynamic Neural Field core used by the black box.

Update rule
-----------
u_{t+1} = u_t + (dt/tau) * ( -u_t + h + (K * a_t) + I_t )
a_t     = sigmoid(u_t; beta, theta)

Kernel
------
K = w_exc * G(sigma_exc) - w_inh * G(sigma_inh)
G is a discrete Gaussian. By default, kernels are normalized to zero-mean and
unit L2-norm ("zero_mean_l2"), which makes gains interpretable and stabilizes
integration across N.

Boundary modes
--------------
- "wrap"    : circular convolution via FFT (fast, typical for DNF on feature rings)
- "reflect" : symmetric padding + valid convolution
- "nearest" : edge padding + valid convolution
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np


Boundary = Literal["wrap", "reflect", "nearest"]
KernelNorm = Literal["none", "zero_mean", "zero_mean_l2"]


@dataclass
class DynamicField:
    size: int
    tau: float
    resting_level: float
    boundary: Boundary = "wrap"
    beta: float = 4.0
    threshold: float = 0.0
    w_exc: float = 1.0
    sigma_exc: float = 2.0
    w_inh: float = 0.5
    sigma_inh: float = 10.0
    kernel_norm: KernelNorm = "zero_mean_l2"
    max_step: float = 0.5  # cap on |du| per step for stability

    def __post_init__(self):
        self.N = int(self.size)
        self._u = np.zeros(self.N, dtype=np.float32)
        self._K = self._build_kernel()
        # Pre-FFT kernel for wrap mode
        self._K_fft = None
        if self.boundary == "wrap":
            self._K_fft = np.fft.rfft(self._K.astype(np.float32))

    # ------------------------- public API -------------------------

    def reset(self, value: float = 0.0):
        """Reset internal state u."""
        self._u.fill(np.float32(value))

    def update(self, I_ext: np.ndarray, dt: float) -> np.ndarray:
        """
        One Euler step. Returns the updated u (view of internal buffer).
        """
        I = np.asarray(I_ext, dtype=np.float32).reshape(-1)
        if I.shape[0] != self.N:
            raise ValueError(f"I_ext length {I.shape[0]} != N={self.N}")

        u = self._u
        a = self._sigmoid(u, self.beta, self.threshold)

        # Lateral interactions
        La = self._convolve(a)

        # du/dt term
        drive = (-u + np.float32(self.resting_level) + La + I)
        du = (dt / self.tau) * drive

        # Optional step limiting for numerical stability (important when dt/tau is large)
        max_abs = float(np.max(np.abs(du)))
        if max_abs > self.max_step > 0.0:
            du = du * (self.max_step / max_abs)

        u += du
        return u  # internal buffer (float32)

    # ------------------------- internals -------------------------

    @staticmethod
    def _sigmoid(u: np.ndarray, beta: float, threshold: float) -> np.ndarray:
        # Stable logistic with float32
        x = np.clip(beta * (u - threshold), -40.0, 40.0, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))

    def _convolve(self, a: np.ndarray) -> np.ndarray:
        """K * a with the configured boundary mode."""
        if self.boundary == "wrap":
            A = np.fft.rfft(a.astype(np.float32))
            y = np.fft.irfft(A * self._K_fft, n=self.N)
            return y.astype(np.float32)

        # Non-wrap: explicit convolution with padding
        k = self._K.astype(np.float32)
        if self.boundary == "reflect":
            pad = "reflect"
        elif self.boundary == "nearest":
            pad = "edge"
        else:
            raise ValueError(f"Unknown boundary mode {self.boundary}")

        r = len(k) // 2
        a_pad = np.pad(a.astype(np.float32), (r, r), mode=pad)
        # correlate with centered kernel (same as convolution since k is symmetric)
        y = np.empty(self.N, dtype=np.float32)
        for i in range(self.N):
            y[i] = np.dot(k, a_pad[i:i+len(k)])
        return y

    # ----- kernel construction -----

    def _build_kernel(self) -> np.ndarray:
        """
        Construct a symmetric difference-of-Gaussians kernel of length N.
        """
        N = self.N
        x = np.arange(N, dtype=np.float32)
        # distance on ring for symmetry (works fine for non-wrap too)
        d = np.minimum(x, N - x).astype(np.float32)

        def gauss(sig):
            sig = max(1e-6, float(sig))
            g = np.exp(-0.5 * (d / sig) ** 2, dtype=np.float32)
            # center at index 0; roll so that center is in the middle for clarity
            return np.roll(g, 0)

        K = (self.w_exc * gauss(self.sigma_exc) -
             self.w_inh * gauss(self.sigma_inh)).astype(np.float32)

        if self.kernel_norm in ("zero_mean", "zero_mean_l2"):
            K = K - np.mean(K, dtype=np.float32)
        if self.kernel_norm == "zero_mean_l2":
            norm = float(np.linalg.norm(K))
            if norm > 0:
                K = K / norm

        return K.astype(np.float32)
