# python_env/adapters/base.py
# -*- coding: utf-8 -*-
"""
Adapter base classes and utilities.

Each dataset plugin must provide a subclass of `BaseAdapter` and register it
with `@registry.register("<adapter_name>")` (see registry.py).

Required adapter interface:
    class MyAdapter(BaseAdapter):
        def iter_sequences(self, split: str):
            yield Sequence(seq_id="unique_id", meta={...}, any_fields=...)

        def iter_frames(self, seq: Sequence):
            for frame in ...:                 # frame can be np.ndarray or any python object
                yield frame

        def frame_to_Iext(self, frame, cfg) -> np.ndarray:
            # Return 1-D I_ext of length cfg["N"], dtype float32.
            return I.astype(np.float32)

Notes
-----
- The runner calls:
    adapter.iter_sequences(split)
    adapter.iter_frames(seq)
    adapter.frame_to_Iext(frame, cfg)
- `frame_to_Iext` MUST return a 1-D float32 numpy array whose length equals cfg["N"].
- Adapters are free to keep any internal state; they should be deterministic
  given the constructor kwargs provided through datasets.yaml.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional
import numpy as np


@dataclass
class Sequence:
    """Lightweight sequence descriptor passed between runner and adapter."""
    seq_id: str
    # Optional path, index, or any dataset-specific metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Sequence(seq_id={self.seq_id!r}, meta_keys={list(self.meta.keys())})"


class BaseAdapter:
    """
    Base dataset adapter. Subclasses should implement the abstract methods.
    """
    def __init__(self, **kwargs):
        # Store kwargs for reproducibility / debugging
        self.kwargs = dict(kwargs)

    # ---------- Abstract API to implement in subclasses ----------

    def iter_sequences(self, split: str) -> Iterable[Sequence]:
        """
        Yield `Sequence` objects for the requested split ('train'|'val'|'test'|...).
        """
        raise NotImplementedError

    def iter_frames(self, seq: Sequence) -> Iterable[Any]:
        """
        Yield raw frames/samples for a `Sequence` (image, event frame, etc.).
        """
        raise NotImplementedError

    def frame_to_Iext(self, frame: Any, cfg: Dict[str, Any]) -> np.ndarray:
        """
        Convert a raw frame to a 1-D external input I_ext of length cfg['N'] (float32).
        """
        raise NotImplementedError

    # ---------- Common helpers for subclasses ----------

    @staticmethod
    def ensure_1d_I(I: np.ndarray, N: int, name: str = "I_ext") -> np.ndarray:
        """
        Validate and coerce an array to 1-D float32 length N.
        """
        I = np.asarray(I, dtype=np.float32).reshape(-1)
        if I.shape[0] != int(N):
            raise ValueError(f"{name} length {I.shape[0]} != N={N}")
        if not np.all(np.isfinite(I)):
            raise ValueError(f"{name} contains NaN/Inf")
        return I

    @staticmethod
    def max_normalize(I: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalize to max=1 (leave zeros as zeros).
        """
        m = float(np.max(I))
        return I if m < eps else (I / m).astype(np.float32)

    @staticmethod
    def clip01(I: np.ndarray) -> np.ndarray:
        """
        Clip into [0,1].
        """
        return np.clip(I, 0.0, 1.0).astype(np.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kwargs={self.kwargs})"
