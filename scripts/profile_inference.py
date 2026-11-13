from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.infer import GREEK_TARGET_LANG  # noqa: E402
from src.asr.model_loader import DEFAULT_MODEL_ID, load_model, resolve_device  # noqa: E402
from src.eval.manifest import load_jsonl_manifest  # noqa: E402
from src.profiling.gpu_metrics import concurrency_test, profile_inference  # noqa: E402
from src.profiling.stats import format_optional_float  # noqa: E402


TARGET_LANG_CHOICES = (GREEK_TARGET_LANG,)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile local ASR inference over a manifest.",
    )
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face model revision (commit hash or tag); recorded in the profile summary.",
    )
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument(
        "--target-lang",
        default=GREEK_TARGET_LANG,
        choices=TARGET_LANG_CHOICES,
        help="Target language hint. Use el for Greek.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark JSON files.")
    parser.add_argument(
        "--device",
        default="auto",
        help='Target device: "auto" (cuda if available, else cpu), "cuda", or "cpu".',
    )
    parser.add_argument(
        "--stream-counts",
        default="1,2,4,8",
        help="Comma-separated local concurrency stream counts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        stream_counts = parse_stream_counts(args.stream_counts)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = load_jsonl_manifest(args.manifest, validate_files=True)
        audio_inputs = [row["audio_filepath"] for row in manifest_rows]

        device = resolve_device(args.device)
        model = load_model(
            model_id_or_path=args.model_id_or_path, device=device, revision=args.revision
        )
        profile_summary = profile_inference(
            model=model,
            audio_inputs=audio_inputs,
            label=args.target_lang,
            target_lang=args.target_lang,
        )
        concurrency_summary = concurrency_test(
            model=model,
            audio_inputs=audio_inputs,
            stream_counts=stream_counts,
            target_lang=args.target_lang,
        )
        profile_summary["model_revision"] = args.revision
        if not device.startswith("cuda"):
            # GPU fields stay empty on a CPU run, even when the machine has a GPU.
            profile_summary["avg_gpu_util_pct"] = None
            for summary in (profile_summary, *concurrency_summary):
                summary["max_memory_allocated_mb"] = None

        _write_json(output_dir / "profile_summary.json", profile_summary)
        _write_json(output_dir / "concurrency_summary.json", concurrency_summary)
        print_profile_summary(profile_summary, concurrency_summary, output_dir)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"Profiling failed: {exc}", file=sys.stderr)
        return 1
    return 0


def parse_stream_counts(value: str) -> list[int]:
    try:
        stream_counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--stream-counts must be a comma-separated list of integers") from exc
    if not stream_counts or any(stream_count <= 0 for stream_count in stream_counts):
        raise ValueError("--stream-counts must contain positive integers")
    return stream_counts


def print_profile_summary(profile_summary: dict, concurrency_summary: list[dict], output_dir: Path) -> None:
    print("Inference profile complete")
    print(f"output_dir: {output_dir}")
    print(f"num_inputs: {profile_summary.get('num_inputs')}")
    print(f"avg_latency_ms: {format_optional_float(profile_summary.get('avg_latency_ms'))}")
    print(f"p95_latency_ms: {format_optional_float(profile_summary.get('p95_latency_ms'))}")
    if concurrency_summary:
        last = concurrency_summary[-1]
        print(
            "max_stream_count: "
            f"{last.get('stream_count')} p95_latency_ms={format_optional_float(last.get('p95_latency_ms'))}"
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
