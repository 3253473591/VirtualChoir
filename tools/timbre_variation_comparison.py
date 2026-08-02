"""Generate a listening comparison for the production timbre variation pipeline.

The default run renders three practical differentiation levels (1, 3 and 5)
with three copies at each level.  That gives nine variants for every source
file while keeping every variant traceable in ``manifest.json``.

Examples:
    python tools/timbre_variation_comparison.py path\\to\\singer.wav
    python tools/timbre_variation_comparison.py path\\to\\wav_folder
    python tools/timbre_variation_comparison.py singer.wav --midi singer.mid

Input audio must meet the application contract: 48 kHz, mono, FLOAT or PCM_32
WAV.  The script deliberately calls ``generate_variations`` rather than
reimplementing any DSP so the comparison is representative of GUI output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virtual_choir.errors import ChoirError
from virtual_choir.models import TimbreVariationConfig
from virtual_choir.timbre_variation import generate_variations


DEFAULT_LEVELS = (1, 3, 5)
LEVEL_LABELS = {
    1: "subtle",
    3: "balanced",
    5: "pronounced",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate three timbre-variation levels with three copies per level "
            "for each 48 kHz mono WAV source."
        ),
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="WAV file(s) or directory/directories containing WAV files.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Empty output directory. Default: timbre_comparison_<timestamp> in the current directory.",
    )
    parser.add_argument(
        "--levels", nargs="+", type=int, default=list(DEFAULT_LEVELS), metavar="LEVEL",
        help="Preset levels to compare (1-5). Default: 1 3 5.",
    )
    parser.add_argument(
        "--copies", type=int, default=3,
        help="Copies rendered for each level and input. Default: 3.",
    )
    parser.add_argument(
        "--voice-style", choices=("popular", "bel_canto", "child"), default="popular",
        help="Articulation profile used by levels 3-5. Default: popular.",
    )
    parser.add_argument(
        "--midi", type=Path, default=None,
        help="Optional MIDI file used for note-aware variation on every input.",
    )
    parser.add_argument(
        "--midi-track-index", type=int, default=None,
        help="Optional MIDI note-track index. Requires --midi.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Discover WAV files recursively inside input directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    levels = _validate_levels(args.levels)
    if not 1 <= args.copies <= 64:
        raise SystemExit("--copies must be between 1 and 64.")
    if args.midi_track_index is not None and args.midi is None:
        raise SystemExit("--midi-track-index requires --midi.")

    midi_path = _resolve_optional_file(args.midi, "MIDI")
    sources = _collect_sources(args.inputs, recursive=args.recursive)
    output_dir = _prepare_output_dir(args.output)

    print(
        f"Comparison: {len(sources)} source(s), levels {', '.join(map(str, levels))}, "
        f"{args.copies} copies per level, voice style {args.voice_style}."
    )
    print(f"Output: {output_dir}")

    manifest: dict[str, object] = {
        "experiment": "timbre_variation_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "levels": list(levels),
        "level_labels": {str(level): LEVEL_LABELS.get(level, f"level_{level}") for level in levels},
        "copies_per_level": args.copies,
        "voice_style": args.voice_style,
        "midi": str(midi_path) if midi_path else None,
        "midi_track_index": args.midi_track_index,
        "sources": [],
    }
    failures = 0
    source_records: list[dict[str, object]] = []
    for index, source_path in enumerate(sources, start=1):
        source_record, source_failures = _render_source(
            index=index,
            source_path=source_path,
            output_dir=output_dir,
            levels=levels,
            copies=args.copies,
            voice_style=args.voice_style,
            midi_path=midi_path,
            midi_track_index=args.midi_track_index,
        )
        source_records.append(source_record)
        failures += source_failures

    manifest["sources"] = source_records
    manifest["failed_jobs"] = failures
    _write_manifest(output_dir, manifest)
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if failures:
        print(f"Completed with {failures} failed level job(s).")
        return 1
    print("Comparison generation complete.")
    return 0


def _validate_levels(values: list[int]) -> tuple[int, ...]:
    levels = tuple(dict.fromkeys(values))
    if not levels:
        raise SystemExit("At least one --levels value is required.")
    if any(level < 1 or level > 5 for level in levels):
        raise SystemExit("--levels values must be between 1 and 5.")
    return levels


def _resolve_optional_file(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} file not found: {resolved}")
    return resolved


def _collect_sources(inputs: list[Path], *, recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".wav":
                raise SystemExit(f"Input is not a WAV file: {path}")
            discovered.append(path)
        elif path.is_dir():
            iterator = path.rglob("*.wav") if recursive else path.glob("*.wav")
            discovered.extend(item.resolve() for item in iterator if item.is_file())
        else:
            raise SystemExit(f"Input path not found: {path}")
    unique = list(dict.fromkeys(discovered))
    if not unique:
        raise SystemExit("No WAV files found.")
    return sorted(unique, key=lambda item: str(item).casefold())


def _prepare_output_dir(requested: Path | None) -> Path:
    if requested is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / f"timbre_comparison_{timestamp}").resolve()
    else:
        output_dir = requested.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _render_source(
    *,
    index: int,
    source_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    copies: int,
    voice_style: str,
    midi_path: Path | None,
    midi_track_index: int | None,
) -> tuple[dict[str, object], int]:
    source_id = f"{index:02d}_{_safe_stem(source_path)}"
    source_dir = output_dir / source_id
    source_record: dict[str, object] = {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_directory": source_id,
        "levels": [],
    }
    level_records: list[dict[str, object]] = []
    failures = 0

    for level in levels:
        label = LEVEL_LABELS.get(level, f"level_{level}")
        level_dir = source_dir / f"level_{level}_{label}"
        print(f"[{index}] {source_path.name} | level {level} ({label})")
        try:
            paths = generate_variations(
                source_path=source_path,
                copy_count=copies,
                output_dir=level_dir,
                config=TimbreVariationConfig.from_preset(level, voice_style),
                progress=_progress_reporter(source_path.name, level),
                midi_path=midi_path,
                midi_track_index=midi_track_index,
            )
            level_records.append({
                "preset_level": level,
                "label": label,
                "voice_style": voice_style,
                "status": "completed",
                "files": [str(path.relative_to(output_dir)) for path in paths],
                "metadata_files": [
                    str(path.with_suffix(".json").relative_to(output_dir)) for path in paths
                ],
            })
        except ChoirError as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
            level_records.append({
                "preset_level": level,
                "label": label,
                "voice_style": voice_style,
                "status": "failed",
                "error_code": exc.code,
                "error": str(exc),
            })
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
            level_records.append({
                "preset_level": level,
                "label": label,
                "voice_style": voice_style,
                "status": "failed",
                "error_code": "UNEXPECTED",
                "error": str(exc),
            })

    source_record["levels"] = level_records
    return source_record, failures


def _progress_reporter(source_name: str, level: int):
    last_percent = -1

    def report(percent: int, message: str) -> None:
        nonlocal last_percent
        if percent == 100 or percent - last_percent >= 10:
            print(f"  {source_name} L{level}: {percent:3d}% {message}")
            last_percent = percent

    return report


def _safe_stem(path: Path) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in path.stem)
    return cleaned.strip("._") or "source"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    target = output_dir / "manifest.json"
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
