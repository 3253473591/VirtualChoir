"""Optional CUDA primitives with a CPU-safe lazy fallback."""

from __future__ import annotations

import os
import threading

import numpy as np

_torch_lock = threading.Lock()
_torch_module = None
_torch_checked = False


def cuda_torch():
    """Return an initialized CUDA-capable torch module, or None."""
    global _torch_checked, _torch_module
    with _torch_lock:
        if _torch_checked:
            return _torch_module
        _torch_checked = True
        try:
            # SciPy and the Windows CUDA torch wheel can both load Intel's
            # OpenMP runtime.  Torch is optional, so retain a CPU fallback if
            # initialization still fails on a particular installation.
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import torch

            if torch.cuda.is_available():
                _torch_module = torch
        except Exception:
            _torch_module = None
        return _torch_module


def fft_convolve(audio: np.ndarray, kernel: np.ndarray) -> np.ndarray | None:
    """Run a full 1-D real convolution on CUDA, or return None when unavailable."""
    torch = cuda_torch()
    if torch is None:
        return None
    try:
        length = len(audio) + len(kernel) - 1
        size = 1 << (length - 1).bit_length()
        with torch.inference_mode():
            source = torch.as_tensor(audio, dtype=torch.float32, device="cuda")
            impulse = torch.as_tensor(kernel, dtype=torch.float32, device="cuda")
            result = torch.fft.irfft(
                torch.fft.rfft(source, n=size) * torch.fft.rfft(impulse, n=size), n=size
            )[:length]
            return result.cpu().numpy()
    except Exception:
        return None


def apply_pitch_cents(f0: np.ndarray, cents: np.ndarray) -> np.ndarray | None:
    """Apply a cents curve to voiced F0 values on CUDA when available."""
    torch = cuda_torch()
    if torch is None or f0.shape != cents.shape:
        return None
    try:
        with torch.inference_mode():
            source = torch.as_tensor(f0, dtype=torch.float64, device="cuda")
            adjustment = torch.as_tensor(cents, dtype=torch.float64, device="cuda")
            result = torch.where(
                source > 0,
                source * torch.exp2(adjustment / 1200.0),
                torch.zeros_like(source),
            )
            return result.cpu().numpy()
    except Exception:
        return None


class CudaLinearSampler:
    """Keep a source waveform on CUDA for repeated linear samples."""

    def __init__(self, source: np.ndarray):
        torch = cuda_torch()
        if torch is None:
            raise RuntimeError("CUDA is unavailable")
        self._torch = torch
        self._source = torch.as_tensor(source, dtype=torch.float32, device="cuda")

    def sample(self, positions: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            values = torch.as_tensor(positions, dtype=torch.float32, device="cuda")
            lower = torch.floor(values).to(torch.long).clamp_(0, len(self._source) - 1)
            upper = (lower + 1).clamp_(0, len(self._source) - 1)
            fraction = values - lower.to(torch.float32)
            result = self._source[lower] * (1.0 - fraction) + self._source[upper] * fraction
            return result.cpu().numpy()


def create_linear_sampler(source: np.ndarray) -> CudaLinearSampler | None:
    try:
        return CudaLinearSampler(source)
    except Exception:
        return None
