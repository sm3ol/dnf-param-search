# python_env/adapters/registry.py
# -*- coding: utf-8 -*-
"""
Simple registry for dataset adapters.

Usage in a plugin:
    from python_env.adapters import registry, base

    @registry.register("ycbsight")
    class YCBSightAdapter(base.BaseAdapter):
        ...

In the runner:
    adapter = registry.get("ycbsight", **dataset_kwargs)
"""

from __future__ import annotations
from typing import Dict, Type
from importlib import import_module

from python_env.adapters.base import BaseAdapter

_ADAPTERS: Dict[str, Type[BaseAdapter]] = {}


def register(name: str):
    """
    Decorator to register a BaseAdapter subclass under `name`.
    """
    def _wrap(cls):
        if not issubclass(cls, BaseAdapter):
            raise TypeError("Only BaseAdapter subclasses can be registered")
        if name in _ADAPTERS:
            # Allow idempotent re-registration with the same class
            if _ADAPTERS[name] is not cls:
                raise KeyError(f"Adapter '{name}' already registered with {_ADAPTERS[name]}")
        _ADAPTERS[name] = cls
        return cls
    return _wrap


def get(name: str, **kwargs) -> BaseAdapter:
    """
    Instantiate an adapter by name. If not present yet, try importing a
    conventional module path: `python_env.adapters.<name>` or `. <name>.loader`.
    """
    if name not in _ADAPTERS:
        # try best-effort lazy import so decorator executes
        tried = []
        for mod in (f"python_env.adapters.{name}.loader", f"python_env.adapters.{name}"):
            try:
                import_module(mod)
                break
            except Exception as e:
                tried.append((mod, e))
        # after attempting imports, check again
        if name not in _ADAPTERS:
            detail = "; ".join([f"{m}: {type(e).__name__}: {e}" for m, e in tried]) or "no modules tried"
            raise KeyError(f"Adapter '{name}' is not registered and could not be imported ({detail}).")

    cls = _ADAPTERS[name]
    return cls(**kwargs)


def list_adapters():
    """
    Return sorted list of registered adapter names.
    """
    return sorted(_ADAPTERS.keys())
