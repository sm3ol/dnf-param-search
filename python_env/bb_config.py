# python_env/bb_config.py
# -*- coding: utf-8 -*-
"""
Config utilities for the DNF black box.

Provides:
    Config (dataclass)
    load_config(path) -> Config
    load_cert(path) -> dict
    config_hash(cfg_or_obj) -> 'sha256:...'
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import yaml


@dataclass
class Config:
    # DNF core
    N: int
    tau: float
    dt: float
    h: float
    beta: float
    theta: float
    w_exc: float
    sigma_exc: float
    w_inh: float
    sigma_inh: float
    # optional
    boundary: str = "wrap"           # 'wrap'|'reflect'|'nearest'
    kernel_norm: str = "zero_mean_l2"
    max_step: float = 0.5            # per-step update cap

    # convenience for repr
    def to_dict(self):
        return asdict(self)


def _yaml_load(path: str | Path):
    txt = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(txt)


def load_config(path: str | Path) -> Config:
    """Load YAML into a Config dataclass."""
    y = _yaml_load(path)
    # allow nesting like {dnf:{...}}; flatten if needed
    src = y.get("dnf", y)

    cfg = Config(
        N=int(src["N"]),
        tau=float(src["tau"]),
        dt=float(src["dt"]),
        h=float(src["h"]),
        beta=float(src["beta"]),
        theta=float(src["theta"]),
        w_exc=float(src["w_exc"]),
        sigma_exc=float(src["sigma_exc"]),
        w_inh=float(src["w_inh"]),
        sigma_inh=float(src["sigma_inh"]),
        boundary=str(src.get("boundary", "wrap")),
        kernel_norm=str(src.get("kernel_norm", "zero_mean_l2")),
        max_step=float(src.get("max_step", 0.5)),
    )
    return cfg


def load_cert(path: str | Path) -> dict:
    """Load certification gates as a plain dictionary."""
    y = _yaml_load(path)
    # support simple YAML or nested under 'gates'
    return y.get("gates", y)


def _stable_json_dump(obj) -> str:
    """Deterministic JSON string for hashing (sort keys & canonical floats)."""
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if hasattr(o, "to_dict"):
                return o.to_dict()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return json.JSONEncoder.default(self, o)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), cls=_Encoder)


def config_hash(cfg_or_obj) -> str:
    """sha256 over a stable JSON representation of the config object or dict."""
    if hasattr(cfg_or_obj, "to_dict"):
        payload = cfg_or_obj.to_dict()
    elif hasattr(cfg_or_obj, "__dict__"):
        payload = dict(cfg_or_obj.__dict__)
    else:
        payload = dict(cfg_or_obj)
    s = _stable_json_dump(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(s).hexdigest()
