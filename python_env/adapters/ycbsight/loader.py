# python_env/adapters/ycbsight/loader.py
# -*- coding: utf-8 -*-
"""
YCBSight adapter (backward compatible)

Supports BOTH schemas in datasets.yaml:

Old (yours, previously working):
--------------------------------
datasets:
  ycbsight_real:
    adapter: ycbsight
    root: "C:/Datasets/YCBSight-Real"
    split_pattern:
      test: "004_sugar_box/gelsight"   # or glob like "test/*"
    mapping:
      N: 100
      projector: "raster_ring"
    adapter_opts:
      intensity_scale_mode: p95
      intensity_scale_floor: 0.0
      intensity_scale_ceil: 3.0
      input_noise_std: 0.0

New (also supported):
---------------------
datasets:
  ycbsight_real:
    adapter: ycbsight
    root: "C:/Datasets/YCBSight-Real"
    split_dirs: { test: "test" }
    frame_glob: "*.png"
    mapper:
      mode: "centroid"
      sigma_bins: 3.0
      intensity_scale_mode: "peak"
      intensity_gain: 1.0
      intensity_floor: 0.0
      intensity_ceiling: 1.0
      input_noise_std: 0.0
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any, Dict, List, Optional
import fnmatch
import numpy as np

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    import imageio as iio  # type: ignore

from ..registry import register
from ..base import BaseAdapter, Sequence
from . import mapper as map1d

# Optional: use projector if provided in datasets.yaml mapping.projector
try:
    from core.projectors import projector_by_name  # raster_ring / angle_ring -> index
except Exception:
    projector_by_name = None  # still works via centroid/argmax mapper

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    """RGB/RGBA → gray float32 in [0,1]."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.299 * r + 0.587 * g + 0.114 * b
    arr = arr.astype("float32", copy=False)
    vmax = float(arr.max()) if arr.size else 1.0
    if vmax > 1.0:
        arr = arr / max(vmax, 1e-6)
    return arr


def _looks_like_glob(s: str) -> bool:
    return any(ch in s for ch in "*?[]")


@register("ycbsight")
class YCBSightAdapter(BaseAdapter):
    def __init__(self, **kwargs):
        """
        kwargs (from datasets.yaml):
            root: path to dataset root (required)
            split_pattern: dict(split -> str | list[str])  [old schema]
            split_dirs:    dict(split -> subdir)           [new schema]
            frame_glob:    glob for frames (default: common image types)
            mapping:       { projector: "raster_ring" | "angle_ring", ... } [old]
            adapter_opts:  legacy mapper options (old)
            mapper:        new-style mapper options (new)
        """
        super().__init__(**kwargs)
        root = kwargs.get("root", None)
        if not root:
            raise ValueError("YCBSightAdapter requires 'root' in datasets.yaml")
        self.root = Path(root)

        # Backward-compat fields
        self.split_pattern: Dict[str, Any] = kwargs.get("split_pattern", {}) or {}
        self.split_dirs: Dict[str, Any] = kwargs.get("split_dirs", {}) or {}

        # Frame glob (optional). If not provided, we scan common img/npy types.
        self.frame_glob: Optional[str] = kwargs.get("frame_glob", None)

        # Old vs new mapper options; normalize to a single dict of options
        adapter_opts = dict(kwargs.get("adapter_opts", {}) or {})
        mapper_new = dict(kwargs.get("mapper", {}) or {})

        # Map legacy names → new
        if "intensity_scale_floor" in adapter_opts and "intensity_floor" not in mapper_new:
            mapper_new["intensity_floor"] = adapter_opts.pop("intensity_scale_floor")
        if "intensity_scale_ceil" in adapter_opts and "intensity_ceiling" not in mapper_new:
            mapper_new["intensity_ceiling"] = adapter_opts.pop("intensity_scale_ceil")
        if "intensity_scale_mode" in adapter_opts and "intensity_scale_mode" not in mapper_new:
            mapper_new["intensity_scale_mode"] = adapter_opts.pop("intensity_scale_mode")
        if "input_noise_std" in adapter_opts and "input_noise_std" not in mapper_new:
            mapper_new["input_noise_std"] = adapter_opts.pop("input_noise_std")

        # Provide sensible defaults if nothing specified
        self.mapper_opts: Dict = {
            "mode": mapper_new.get("mode", "centroid"),
            "sigma_bins": float(mapper_new.get("sigma_bins", 3.0)),
            "intensity_scale_mode": mapper_new.get("intensity_scale_mode", "peak"),
            "intensity_gain": float(mapper_new.get("intensity_gain", 1.0)),
            "intensity_floor": float(mapper_new.get("intensity_floor", 0.0)),
            "intensity_ceiling": float(mapper_new.get("intensity_ceiling", 1.0)),
            "input_noise_std": float(mapper_new.get("input_noise_std", 0.0)),
        }

        # Optional projector (old schema): mapping.projector -> choose center index
        mapping = dict(kwargs.get("mapping", {}) or {})
        self.projector_name: Optional[str] = mapping.get("projector", None)

    # ----------------- discovery helpers -----------------

    def _list_frames(self, seq_dir: Path) -> List[Path]:
        if self.frame_glob:
            files = sorted(seq_dir.glob(self.frame_glob))
        else:
            # default: common imgs + npy in THIS folder only
            files = []
            for ext in IMG_EXTS:
                files.extend(seq_dir.glob(f"*{ext}"))
            files = sorted(files)
        return [f for f in files if f.is_file()]

    def _discover_by_split_pattern(self, split: str) -> List[Path]:
        """
        Support old schema: split_pattern[split] can be a subfolder or a glob.
        Returns a list of *sequence directories*.
        """
        pat = self.split_pattern.get(split)
        if pat is None:
            return []
        pats = [pat] if isinstance(pat, (str, Path)) else list(pat)
        seq_dirs: List[Path] = []
        for p in pats:
            p = str(p)
            if _looks_like_glob(p):
                # glob under root
                for m in self.root.glob(p):
                    if m.is_dir():
                        seq_dirs.append(m)
            else:
                # direct subpath under root
                d = (self.root / p)
                if d.is_dir():
                    seq_dirs.append(d)

        # If a listed directory has no frames but has child dirs,
        # treat each child dir with frames as a separate sequence.
        out: List[Path] = []
        for d in seq_dirs:
            frames = self._list_frames(d)
            if frames:
                out.append(d)
            else:
                children = [c for c in d.iterdir() if c.is_dir()]
                # keep only those children that contain frames
                for c in children:
                    if self._list_frames(c):
                        out.append(c)
        # de-dup and sort
        return sorted(list(dict.fromkeys(out)), key=lambda p: p.as_posix())

    def _discover_by_split_dir(self, split: str) -> List[Path]:
        """
        New schema: split_dirs[split] -> subdir name. If that dir contains
        subdirs, each is a sequence; otherwise, the dir itself is a single sequence.
        """
        if split not in self.split_dirs:
            return []
        split_dir = self.root / self.split_dirs[split]
        if not split_dir.exists():
            return []
        # If it has child dirs, treat each as a sequence; else, itself.
        children = [p for p in split_dir.iterdir() if p.is_dir()]
        if children:
            return sorted(children, key=lambda p: p.name)
        return [split_dir]

    # ----------------- BaseAdapter API -----------------

    def iter_sequences(self, split: str) -> Iterable[Sequence]:
        # Old schema first (your case)
        seq_dirs = self._discover_by_split_pattern(split)
        # Fallback to new schema
        if not seq_dirs:
            seq_dirs = self._discover_by_split_dir(split)
        # Final fallback: allow passing an ABSOLUTE path via --split
        if not seq_dirs:
            sp = Path(split)
            if sp.exists() and sp.is_dir():
                seq_dirs = [sp]

        if not seq_dirs:
            # Helpful error with suggestions
            kids = ", ".join(sorted(p.name for p in self.root.iterdir() if p.is_dir()))
            raise FileNotFoundError(
                f"No sequences for split='{split}'.\n"
                f"root='{self.root}'\n"
                f"Try fixing datasets.yaml (split_pattern or split_dirs). "
                f"Subfolders under root: [{kids}]"
            )

        for sd in seq_dirs:
            frames = self._list_frames(sd)
            if not frames:
                # skip empty dirs
                continue
            meta = {"dir": str(sd), "num_frames": len(frames)}
            meta["frame_paths"] = [str(p) for p in frames]
            yield Sequence(seq_id=sd.name, meta=meta)

    def iter_frames(self, seq: Sequence) -> Iterable[Any]:
        frame_paths = seq.meta.get("frame_paths")
        if not frame_paths:
            # safety fallback
            d = Path(seq.meta.get("dir", ""))
            frame_paths = [str(p) for p in self._list_frames(d)]
        for fp in frame_paths:
            p = Path(fp)
            if p.suffix.lower() == ".npy":
                frame = np.load(p)
            else:
                frame = iio.imread(p)
            yield _to_gray_f32(frame)

    def frame_to_Iext(self, frame: Any, cfg: Dict) -> np.ndarray:
        """
        Convert a single grayscale frame (H×W in [0,1]) to a 1-D I_ext of length N.

        If datasets.yaml specifies mapping.projector, we use it to pick a center
        index on the 1-D ring and place a Gaussian bump (legacy behavior).
        Otherwise we use the new mapper (centroid/argmax + amplitude scaling).
        """
        frame = np.asarray(frame, dtype="float32")
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame, got shape {frame.shape}")
        N = int(cfg["N"])

        # Legacy projector path?
        if self.projector_name and projector_by_name is not None:
            try:
                proj = projector_by_name(self.projector_name)
            except Exception:
                proj = None
            if proj is not None:
                # center index from projector, then build Gaussian bump
                idx = float(proj(frame, N))
                gauss = map1d._gaussian_1d(center_idx=idx, N=N, sigma_bins=float(self.mapper_opts["sigma_bins"]))
                # amplitude via mapper's robust scaling
                amp = map1d._amplitude_from_frame(
                    frame,
                    mode=self.mapper_opts["intensity_scale_mode"],
                    gain=float(self.mapper_opts["intensity_gain"]),
                    floor=float(self.mapper_opts["intensity_floor"]),
                    ceil=float(self.mapper_opts["intensity_ceiling"]),
                )
                I = (amp * gauss).astype("float32")
                noise = float(self.mapper_opts.get("input_noise_std", 0.0))
                if noise > 0.0:
                    I = I + np.random.normal(0.0, noise, size=I.shape).astype("float32")
                return I

        # Default new-style mapping (centroid/argmax)
        return map1d.frame_to_Iext_1d(frame, N=N, **self.mapper_opts)
