"""Shared YAML config loader for every entry point.

Every string value is passed through `os.path.expanduser(os.path.expandvars(...))`,
so a config can reference `${DATA_ROOT}`/`~` instead of a machine-specific
absolute path -- set the env var per machine, keep the config portable.
"""

from __future__ import annotations

import os

import yaml


def _expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str) -> dict:
    with open(path) as f:
        return _expand(yaml.safe_load(f))
