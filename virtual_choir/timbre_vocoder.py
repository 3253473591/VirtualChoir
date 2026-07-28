"""OpenVPI PC-NSF-HiFiGAN analysis and synthesis boundary.

The variation pipeline supplies an F0 curve and receives a 48 kHz waveform.
Keeping the heavyweight model lifecycle here prevents UI and variation-policy
code from depending on OpenVPI implementation details.
"""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .bootstrap import vocoder_paths
from .errors import ChoirError


MODEL_SAMPLE_RATE = 44_100
MODEL_HOP_SAMPLES = 512
MODEL_FFT_SAMPLES = 2048
MODEL_MEL_BINS = 128

_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_MODEL_SOURCE = _ROOT / "tools" / "vendor" / "SingingVocoders" / "models" / "nsf_HiFigan" / "models.py"
_LEGACY_CHECKPOINT = _ROOT / "tools" / "vendor" / "SingingVocoders" / "checkpoints" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt"
_VOCODER_LOCK = threading.Lock()
_VOCODER: _PcNsfHifiGan | None = None


@dataclass(frozen=True)
class VocoderAnalysis:
    """Shared conditioning features for all copies derived from one source."""

    f0: np.ndarray
    mel: object


@dataclass(frozen=True)
class _PcNsfHifiGan:
    generator: object
    torch: object
    device: object


def analyze_voice(source: np.ndarray, cancel_event: threading.Event | None = None) -> VocoderAnalysis:
    """Extract voiced CREPE F0 and the official 44.1 kHz OpenVPI Mel input."""
    _check_cancel(cancel_event)
    try:
        import librosa
        import torch
        import torchcrepe

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_audio = signal.resample_poly(
            source.astype(np.float64), 147, 160, window=("kaiser", 8.6)
        ).astype(np.float32)
        tensor = torch.from_numpy(model_audio).unsqueeze(0).to(device)
        with torch.inference_mode():
            crepe_f0, periodicity = torchcrepe.predict(
                tensor, MODEL_SAMPLE_RATE, MODEL_HOP_SAMPLES, 50, 1100,
                model="full" if device.type == "cuda" else "tiny",
                batch_size=2048, device=device, return_periodicity=True,
            )
            mel = _make_mel(torch, librosa, tensor)
        f0, periodicity = _align_f0_to_mel(
            np.asarray(crepe_f0.squeeze(0).cpu(), dtype=np.float64),
            np.asarray(periodicity.squeeze(0).cpu(), dtype=np.float64), int(mel.shape[-1]),
        )
        f0 = np.where((periodicity >= 0.45) & (f0 > 0), f0, 0.0)
    except Exception as exc:
        raise ChoirError("RENDER_FAILED", f"CREPE/PC-NSF-HiFiGAN 音频分析失败：{exc}") from exc
    _check_cancel(cancel_event)
    return VocoderAnalysis(f0, mel.detach())


def synthesize(
    analysis: VocoderAnalysis, f0: np.ndarray, cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """Render a 48 kHz waveform from a Mel spectrum and external F0 curve."""
    _check_cancel(cancel_event)
    try:
        vocoder = _load()
        f0_tensor = vocoder.torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).to(vocoder.device)
        with _VOCODER_LOCK, vocoder.torch.inference_mode():
            output = vocoder.generator(analysis.mel.to(vocoder.device), f0_tensor)
        output = output.squeeze().detach().cpu().numpy()
        return signal.resample_poly(
            output.astype(np.float64), 160, 147, window=("kaiser", 8.6)
        ).astype(np.float32)
    except ChoirError:
        raise
    except Exception as exc:
        raise ChoirError("RENDER_FAILED", f"PC-NSF-HiFiGAN 重合成失败：{exc}") from exc


def _load() -> _PcNsfHifiGan:
    global _VOCODER
    if _VOCODER is not None:
        return _VOCODER
    model_source, checkpoint = vocoder_paths()
    # Keep older manually provisioned installations working.
    if not model_source.is_file() and _LEGACY_MODEL_SOURCE.is_file():
        model_source = _LEGACY_MODEL_SOURCE
    if not checkpoint.is_file() and _LEGACY_CHECKPOINT.is_file():
        checkpoint = _LEGACY_CHECKPOINT
    if not checkpoint.is_file() or not model_source.is_file():
        raise ChoirError(
            "RENDER_DEPENDENCY_MISSING",
            "缺少 OpenVPI PC-NSF-HiFiGAN 权重；请重新启动以完成工作目录 models/SingingVocoders 的初始化。",
        )
    with _VOCODER_LOCK:
        if _VOCODER is not None:
            return _VOCODER
        try:
            import torch

            spec = importlib.util.spec_from_file_location("virtual_choir_openvpi_nsf_hifigan", model_source)
            if spec is None or spec.loader is None:
                raise RuntimeError("无法加载 OpenVPI 模型源码")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = module.AttrDict({
                "mini_nsf": True,
                "noise_sigma": 0.0,
                "upsample_rates": [8, 8, 2, 2, 2],
                "upsample_kernel_sizes": [16, 16, 4, 4, 4],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "resblock": "1",
                "sampling_rate": MODEL_SAMPLE_RATE,
                "num_mels": MODEL_MEL_BINS,
            })
            generator = module.Generator(config)
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state_dict = {
                key.removeprefix("generator."): value
                for key, value in checkpoint_data["state_dict"].items()
                if key.startswith("generator.")
            }
            missing, unexpected = generator.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"权重不匹配：missing={missing[:3]}, unexpected={unexpected[:3]}")
            generator.remove_weight_norm()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _VOCODER = _PcNsfHifiGan(generator.eval().to(device), torch, device)
            return _VOCODER
        except ChoirError:
            raise
        except Exception as exc:
            raise ChoirError("RENDER_DEPENDENCY_MISSING", f"无法加载 PC-NSF-HiFiGAN：{exc}") from exc


def _make_mel(torch, librosa, waveform):
    mel_basis = torch.from_numpy(librosa.filters.mel(
        sr=MODEL_SAMPLE_RATE, n_fft=MODEL_FFT_SAMPLES, n_mels=MODEL_MEL_BINS,
        fmin=40, fmax=16_000,
    )).float().to(waveform.device)
    padding = MODEL_FFT_SAMPLES - MODEL_HOP_SAMPLES
    padded = torch.nn.functional.pad(
        waveform.unsqueeze(1), (padding // 2, (padding + 1) // 2), mode="reflect"
    ).squeeze(1)
    spec = torch.stft(
        padded, MODEL_FFT_SAMPLES, hop_length=MODEL_HOP_SAMPLES,
        win_length=MODEL_FFT_SAMPLES,
        window=torch.hann_window(MODEL_FFT_SAMPLES, device=waveform.device),
        center=False, normalized=False, onesided=True, return_complex=True,
    ).abs()
    return torch.log(torch.clamp(torch.matmul(mel_basis, spec), min=1e-5))


def _align_f0_to_mel(f0: np.ndarray, periodicity: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    if frame_count <= 0 or not len(f0):
        return np.zeros(frame_count), np.zeros(frame_count)
    indices = np.clip(np.rint(np.linspace(0, len(f0) - 1, frame_count)).astype(np.intp), 0, len(f0) - 1)
    return f0[indices], periodicity[indices]


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ChoirError("RENDER_CANCELLED")
