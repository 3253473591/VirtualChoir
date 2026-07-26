"""Standalone OpenVPI PC-NSF-HiFiGAN F0 redraw listening experiment.

This script is intentionally outside the application pipeline. It accepts one
WAV file, reconstructs it with the official 44.1 kHz checkpoint, and writes a
baseline plus the project's five redraw levels beside the source file.

Example (Python 3.11 environment):
    D:\\anaconda3\\envs\\py311_env\\python.exe tools\\pc_nsf_hifigan_ab_experiment.py input.wav

The checkpoint is CC BY-NC-SA 4.0. See the vendor checkpoint NOTICE files for
the attribution and distribution terms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


SOURCE_SAMPLE_RATE = 48_000
MODEL_SAMPLE_RATE = 44_100
HOP_SIZE = 512
N_FFT = 2048
N_MELS = 128
FMIN = 40
FMAX = 16_000
CHECKPOINT = (
    Path(__file__).resolve().parent
    / "vendor"
    / "SingingVocoders"
    / "checkpoints"
    / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt"
)

# Original/rebuilt F0 mix: the rebuilt component is 50/60/70/80/95 percent.
REDRAW_MIX = {1: 0.50, 2: 0.60, 3: 0.70, 4: 0.80, 5: 0.95}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PC-NSF-HiFiGAN baseline and five F0-redraw variants."
    )
    parser.add_argument("input_wav", type=Path, help="Input WAV file.")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Inference device; auto prefers CUDA.",
    )
    parser.add_argument(
        "--crepe-model", choices=("full", "tiny"), default=None,
        help="CREPE model; auto uses full on CUDA and tiny on CPU.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Default is deterministic from the input file.",
    )
    parser.add_argument(
        "--proxy", default="http://127.0.0.1:7890",
        help="Proxy for CREPE model downloads; empty string disables it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_wav.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input WAV not found: {input_path}")
    if input_path.suffix.lower() != ".wav":
        raise SystemExit("Input must be a WAV file.")
    if args.proxy:
        os.environ.setdefault("HTTP_PROXY", args.proxy)
        os.environ.setdefault("HTTPS_PROXY", args.proxy)

    torch, sf, resample_poly, predict, Generator, AttrDict = _load_dependencies()
    device = _resolve_device(torch, args.device)
    audio, source_rate = sf.read(input_path, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise SystemExit("Input WAV is empty or contains non-finite samples.")
    # A mono sum keeps this tool useful with ordinary stereo exports while the
    # vocoder itself still receives the single channel it was trained for.
    audio = np.asarray(audio, dtype=np.float32).mean(axis=1)
    model_audio = _resample(audio, int(source_rate), MODEL_SAMPLE_RATE, resample_poly)
    model_audio = np.clip(model_audio, -1.0, 1.0)

    print(f"Loading PC-NSF-HiFiGAN on {device}...")
    generator = _load_generator(torch, Generator, AttrDict, device)
    mel = _make_mel(torch, model_audio, device)
    crepe_model = args.crepe_model or ("full" if device.type == "cuda" else "tiny")
    print(f"Extracting CREPE F0 ({crepe_model})...")
    original_f0, periodicity = _extract_f0(
        torch, predict, model_audio, crepe_model, device
    )
    frame_count = int(mel.shape[-1])
    original_f0, periodicity = _align_f0(original_f0, periodicity, frame_count)
    voiced = (original_f0 > 0) & (periodicity >= 0.45)
    original_f0 = np.where(voiced, original_f0, 0.0).astype(np.float32)

    output_dir = input_path.parent / f"{input_path.stem}_pc_nsf_hifigan_ab"
    output_dir.mkdir(exist_ok=True)
    seed_base = args.seed
    if seed_base is None:
        seed_base = int.from_bytes(
            hashlib.sha256(input_path.read_bytes()).digest()[:8], "big"
        )

    records = []
    records.append(_render(
        torch, sf, resample_poly, generator, mel, original_f0, device,
        input_path, output_dir, "baseline", 0, 0.0, seed_base,
        target_samples=round(len(audio) * SOURCE_SAMPLE_RATE / int(source_rate)),
    ))
    for level, mix in REDRAW_MIX.items():
        redraw = _make_redraw_line(len(original_f0), voiced, seed_base + level)
        variant_f0 = _blend_f0(original_f0, redraw, mix, voiced)
        records.append(_render(
            torch, sf, resample_poly, generator, mel, variant_f0, device,
            input_path, output_dir, f"level_{level}", level, mix,
            seed_base + level,
            target_samples=round(len(audio) * SOURCE_SAMPLE_RATE / int(source_rate)),
        ))

    metadata = {
        "engine": "openvpi_pc_nsf_hifigan_f0_redraw_v1",
        "checkpoint": CHECKPOINT.name,
        "checkpoint_license": "CC BY-NC-SA 4.0",
        "source_sample_rate_hz": int(source_rate),
        "model_sample_rate_hz": MODEL_SAMPLE_RATE,
        "output_sample_rate_hz": SOURCE_SAMPLE_RATE,
        "device": str(device),
        "crepe_model": crepe_model,
        "crepe_periodicity_threshold": 0.45,
        "mel_frames": frame_count,
        "voiced_frame_ratio": float(voiced.mean()) if len(voiced) else 0.0,
        "redraw_definition": (
            "Smooth random cents contour on voiced frames; no high-rate jitter "
            "or forced interpolation through unvoiced gaps."
        ),
        "variants": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} files to: {output_dir}")
    return 0


def _load_dependencies():
    try:
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly
        import torchcrepe
    except ImportError as exc:
        raise SystemExit(f"Missing dependency in py311_env: {exc}") from exc

    model_dir = Path(__file__).resolve().parent / "vendor" / "SingingVocoders"
    model_file = model_dir / "models" / "nsf_HiFigan" / "models.py"
    if not model_file.is_file() or not CHECKPOINT.is_file():
        raise SystemExit(
            "OpenVPI source/checkpoint is missing under tools/vendor/SingingVocoders."
        )
    spec = importlib.util.spec_from_file_location("openvpi_nsf_hifigan_models", model_file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import model source: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return torch, sf, resample_poly, torchcrepe.predict, module.Generator, module.AttrDict


def _resolve_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable in py311_env.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def _load_generator(torch, Generator, AttrDict, device):
    config = AttrDict({
        # The 2025.02 checkpoint is the official fast/mini NSF variant; its
        # state dict contains source_conv rather than m_source/noise_convs.
        "mini_nsf": True,
        "upsample_rates": [8, 8, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock": "1",
        "noise_sigma": 0.0,
        "sampling_rate": MODEL_SAMPLE_RATE,
        "num_mels": N_MELS,
        "hop_size": HOP_SIZE,
    })
    generator = Generator(config)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    state_dict = {
        key.removeprefix("generator."): value
        for key, value in state_dict.items()
        if key.startswith("generator.")
    }
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"Checkpoint/model mismatch; missing={list(missing)[:5]}, "
            f"unexpected={list(unexpected)[:5]}"
        )
    generator.remove_weight_norm()
    return generator.eval().to(device)


def _make_mel(torch, audio: np.ndarray, device):
    from librosa.filters import mel as librosa_mel_fn

    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    mel_basis = torch.from_numpy(librosa_mel_fn(
        sr=MODEL_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
    )).float().to(device)
    window = torch.hann_window(N_FFT, device=device)
    pad_left = (N_FFT - HOP_SIZE) // 2
    pad_right = (N_FFT - HOP_SIZE + 1) // 2
    waveform = torch.nn.functional.pad(
        waveform.unsqueeze(1), (pad_left, pad_right), mode="reflect"
    ).squeeze(1)
    spec = torch.stft(
        waveform, N_FFT, hop_length=HOP_SIZE, win_length=N_FFT,
        window=window, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    ).abs()
    return torch.log(torch.clamp(torch.matmul(mel_basis, spec), min=1e-5))


def _extract_f0(torch, predict, audio, model_name, device):
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.inference_mode():
        f0, periodicity = predict(
            waveform, MODEL_SAMPLE_RATE, hop_length=HOP_SIZE,
            fmin=50.0, fmax=1100.0, model=model_name,
            return_periodicity=True, device=device,
        )
    return (
        f0.squeeze(0).detach().cpu().numpy().astype(np.float32),
        periodicity.squeeze(0).detach().cpu().numpy().astype(np.float32),
    )


def _align_f0(f0, periodicity, target_length):
    if len(f0) == target_length:
        return f0, periodicity
    if not len(f0):
        return np.zeros(target_length, dtype=np.float32), np.zeros(target_length, dtype=np.float32)
    source_index = np.linspace(0.0, 1.0, len(f0))
    target_index = np.linspace(0.0, 1.0, target_length)
    indices = np.clip(np.rint(target_index * (len(f0) - 1)).astype(int), 0, len(f0) - 1)
    return f0[indices], periodicity[indices]


def _make_redraw_line(length, voiced, seed):
    rng = np.random.default_rng(seed)
    if length == 0:
        return np.zeros(0, dtype=np.float32)
    # Low-rate control points produce broad human-like drift rather than FM.
    control_step = max(5, round(0.18 * MODEL_SAMPLE_RATE / HOP_SIZE))
    control_count = max(2, int(np.ceil(length / control_step)) + 1)
    controls = rng.normal(0.0, 32.0, control_count).astype(np.float32)
    control_x = np.linspace(0.0, length - 1, control_count)
    line = np.interp(np.arange(length), control_x, controls).astype(np.float32)
    # Remove DC per contiguous voiced segment so each note is not systematically
    # transposed while preserving the original F0 contour as the baseline.
    result = np.zeros(length, dtype=np.float32)
    start = None
    for index in range(length + 1):
        is_voiced = index < length and bool(voiced[index])
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            segment = line[start:index]
            result[start:index] = segment - float(segment.mean())
            start = None
    return result


def _blend_f0(original_f0, redraw, mix, voiced):
    result = original_f0 * np.power(2.0, redraw * mix / 1200.0)
    return np.where(voiced, result, 0.0).astype(np.float32)


def _render(torch, sf, resample_poly, generator, mel, f0, device,
            input_path, output_dir, name, level, mix, seed, target_samples):
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) & 0x7FFFFFFF)
    f0_tensor = torch.from_numpy(f0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = generator(mel, f0_tensor).squeeze().detach().cpu().numpy()
    output_48k = _resample(output, MODEL_SAMPLE_RATE, SOURCE_SAMPLE_RATE, resample_poly)
    # Mel framing can add a short tail; make every A/B file duration-match the
    # source after conversion to the requested 48 kHz output rate.
    if len(output_48k) < target_samples:
        output_48k = np.pad(output_48k, (0, target_samples - len(output_48k)))
    output_48k = output_48k[:target_samples]
    output_path = output_dir / f"{input_path.stem}_pc_nsf_hifigan_{name}.wav"
    sf.write(output_path, np.clip(output_48k, -1.0, 1.0), SOURCE_SAMPLE_RATE, subtype="FLOAT")
    return {
        "level": level,
        "file": output_path.name,
        "redraw_mix": mix,
        "samples": int(len(output_48k)),
        "seed": int(seed),
    }


def _resample(audio, source_rate, target_rate, resample_poly):
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = np.gcd(source_rate, target_rate)
    result = resample_poly(
        np.asarray(audio, dtype=np.float64), target_rate // divisor, source_rate // divisor
    )
    return np.asarray(result, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
