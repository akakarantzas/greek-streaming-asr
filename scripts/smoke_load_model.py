"""Defensive real-load smoke script for the Nemotron 3.5 ASR model.

This script intentionally does not implement inference, transcription, audio
loading, streaming, benchmarking, profiling, or fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.model_loader import (  # noqa: E402
    DEFAULT_MODEL_ID,
    get_gpu_memory_allocated_mb,
    load_model,
    resolve_device,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-load the Nemotron 3.5 ASR model through the project loader.",
    )
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Target device: "auto" (cuda if available, else cpu), "cuda", or "cpu".',
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision, if supported by the installed NeMo API.",
    )
    return parser


def build_troubleshooting_message(error: BaseException) -> str:
    return "\n".join(
        (
            f"Model load failed: {error}",
            "",
            "Troubleshooting:",
            "- Confirm NeMo ASR is installed in the active Python environment.",
            '- If using device="cuda", confirm the NVIDIA driver, CUDA-compatible PyTorch wheel, and GPU visibility.',
            "- Confirm Hugging Face/NVIDIA access for nvidia/nemotron-3.5-asr-streaming-0.6b.",
            "- Try a local .nemo checkpoint path with --model-id-or-path if available.",
            "- If running without DGX Spark, use this script only as a CLI/import smoke test until proper hardware is available.",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print(f"Selected model ID/path: {args.model_id_or_path}")
    device = resolve_device(args.device)
    print(f"Selected device: {device}")
    if args.revision:
        print(f"Selected revision: {args.revision}")

    before_mb = get_gpu_memory_allocated_mb()  # None without torch or CUDA
    if before_mb is not None:
        print(f"GPU memory before loading: {before_mb:.2f} MB")

    try:
        model = load_model(
            model_id_or_path=args.model_id_or_path,
            device=device,
            revision=args.revision,
        )
    except Exception as exc:
        print(build_troubleshooting_message(exc), file=sys.stderr)
        return 1

    after_mb = get_gpu_memory_allocated_mb()
    if after_mb is not None:
        print(f"GPU memory after loading: {after_mb:.2f} MB")
    print(f"Loaded model class: {type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
