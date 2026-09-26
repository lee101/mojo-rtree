"""ctypes loader for the Mojo R-tree kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "rtree.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-rtree.so")

I = ctypes.c_int64

_SIGNATURES = {
    "mrt_build": ([I] * 9, I),
    "mrt_intersection": ([I] * 10, I),
    "mrt_intersection_count": ([I] * 9, I),
    "mrt_intersection_counts": ([I] * 11, None),
    "mrt_intersection_fill": ([I] * 12, None),
    "mrt_nearest": ([I] * 13, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("Mojo shared library is missing; run `pixi run build`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_instance: ctypes.CDLL | None = None



def lib() -> ctypes.CDLL:
    global _instance
    if _instance is None:
        _instance = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_instance, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _instance



def addr(array: np.ndarray, dtype: np.dtype) -> int:
    """Return an address only for a non-empty, ABI-compatible NumPy buffer."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != expected:
        raise TypeError(f"FFI buffer must have dtype {expected}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty")
    return int(array.ctypes.data)
