"""Choose an ISA-specific executable before measurement, never per probe."""
from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


def supports_sve_bitperm() -> bool:
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    try:
        getauxval = ctypes.CDLL(None).getauxval
        getauxval.argtypes = [ctypes.c_ulong]
        getauxval.restype = ctypes.c_ulong
        # Linux arm64 asm/hwcap.h: kernel-advertised features, not build-host flags.
        return bool(getauxval(16) & (1 << 22)) and (getauxval(26) & 0x12) == 0x12
    except (AttributeError, OSError):
        return False


def select_benchmark_binary(base: Path) -> Path:
    mode = os.environ.get("BENCH_DIVA_BACKEND", "auto")
    if mode not in {"auto", "portable", "sve2"}:
        raise ValueError("BENCH_DIVA_BACKEND must be auto, portable or sve2")
    optimized = base.with_name(base.name + "_sve2")
    available = optimized.is_file() and os.access(optimized, os.X_OK)
    supported = supports_sve_bitperm()
    if mode == "sve2" and not (available and supported):
        raise ValueError("Diva SVE2 requires both an optimized binary and kernel SVE2/bitperm support")
    return (optimized if mode != "portable" and available and supported else base).resolve()


def binary_backend(binary: Path) -> str:
    if binary.name.endswith("_sve2"):
        return "sve2-bitperm"
    return "portable-arm" if platform.machine().lower() in {"aarch64", "arm64"} else "native"
