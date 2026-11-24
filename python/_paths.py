"""
_paths.py
---------
Shared resource path resolver so modules work uniformly whether invoked:
1. Directly inside a cloned repository checkout (`python python/<script>.py`)
2. As an installed pip/wheel package (`import rabi_crop_doctor`, reading shared data_files)
"""

from __future__ import annotations

import os
import sys


def get_project_root() -> str:
    """Return the local repository root (parent directory of python/)."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def resolve_asset_path(*parts: str) -> str:
    """
    Locate a bundled asset (`data/...` or `models/...`) across source checkout,
    pip wheel `sys.prefix/share/rabi_crop_doctor`, or current working directory.
    """
    candidates = [
        os.path.join(get_project_root(), *parts),
        os.path.join(sys.prefix, "share", "rabi_crop_doctor", *parts),
        os.path.join(os.getcwd(), *parts),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    # Default to repository path so downstream error messages cite the expected file location
    return candidates[0]
