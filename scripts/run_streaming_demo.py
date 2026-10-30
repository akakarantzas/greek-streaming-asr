from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.infer import GREEK_TARGET_LANG  # noqa: E402
from src.asr.model_loader import (  # noqa: E402
    DEFAULT_MODEL_ID,
    get_gpu_name,
    load_model,
    resolve_device,
)
from src.asr.streaming_asr import ChunkedASRSession  # noqa: E402
from src.audio.microphone import DEFAULT_SAMPLE_RATE, MicrophoneAudioStream  # noqa: E402
from src.profiling.stats import mean as _mean  # noqa: E402
from src.profiling.stats import percentile  # noqa: E402


LANGUAGE_CHOICES = ("el",)
TARGET_LANG_CHOICES = (GREEK_TARGET_LANG, "auto")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a terminal microphone streaming ASR demo.",
    )
    parser.add_argument(
        "--model-id-or-path",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local .nemo checkpoint path.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Microphone chunk size in milliseconds.",
    )
    parser.add_argument(
        "--inference-window-ms",
        type=int,
        default=1000,
        help="How often to run ASR over the rolling buffer, in milliseconds.",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGE_CHOICES,
        required=True,
        help="Display language label.",
    )
    parser.add_argument(
        "--target-lang",
        choices=TARGET_LANG_CHOICES,
        default=GREEK_TARGET_LANG,
        help="Target language hint for ASR when supported.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Target device: "auto" (cuda if available, else cpu), "cuda", or "cpu".',
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional session limit for testing.",
    )
    return parser


def run_demo(args: argparse.Namespace) -> int:
    chunk_samples = chunk_ms_to_samples(args.chunk_ms, DEFAULT_SAMPLE_RATE)
    inference_window_s = args.inference_window_ms / 1000

    device = resolve_device(args.device)
    model = load_model(model_id_or_path=args.model_id_or_path, device=device)
    session = ChunkedASRSession(
        model=model,
        sample_rate=DEFAULT_SAMPLE_RATE,
        target_lang=args.target_lang,
        inference_interval_s=inference_window_s,
        log_events=False,
    )
    print_startup_header(
        model_id_or_path=args.model_id_or_path,
        target_lang=args.target_lang,
        chunk_ms=args.chunk_ms,
        inference_window_ms=args.inference_window_ms,
        language=args.language,
        language_readiness=session.language_readiness,
        readiness_warnings=session.session_warnings,
        device=device,
        gpu_name=get_gpu_name(),
        mode=session.mode,
    )

    events: list[dict] = []
    microphone = MicrophoneAudioStream(chunk_size=chunk_samples)
    start = time.perf_counter()
    try:
        with microphone:
            for chunk in microphone:
                if args.max_seconds is not None and time.perf_counter() - start >= args.max_seconds:
                    break
                event = session.process_chunk(chunk)
                if event is None:
                    continue
                events.append(event)
                print_partial_event(event)
    except KeyboardInterrupt:
        print("\nStreaming demo interrupted by user.")
    finally:
        total_runtime_s = time.perf_counter() - start
        print()
        dropped_audio_s = microphone.dropped_chunks * chunk_samples / DEFAULT_SAMPLE_RATE
        print_session_summary(summarize_session(events, total_runtime_s, dropped_audio_s))
    return 0


def summarize_session(
    events: list[dict],
    total_runtime_s: float,
    dropped_audio_s: float = 0.0,
) -> dict:
    latencies = [float(event["latency_ms"]) for event in events if event.get("latency_ms") is not None]
    rtfs = [float(event["rtf"]) for event in events if event.get("rtf") is not None]
    return {
        "total_runtime_s": total_runtime_s,
        "inference_windows": len(events),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "avg_rtf": _mean(rtfs),
        "dropped_audio_s": dropped_audio_s,
    }


def print_startup_header(
    model_id_or_path: str,
    target_lang: str,
    chunk_ms: int,
    inference_window_ms: int,
    language: str,
    language_readiness: str,
    readiness_warnings: list[str],
    device: str,
    gpu_name: str | None,
    mode: str,
) -> None:
    print("Streaming ASR demo")
    print(f"model: {model_id_or_path}")
    print(f"target_lang: {target_lang}")
    print(f"chunk_ms: {chunk_ms}")
    print(f"inference_window_ms: {inference_window_ms}")
    print(f"language: {language}")
    print(f"language_readiness: {language_readiness}")
    for warning in readiness_warnings:
        print(f"warning: {warning}")
    print(f"device: {device}")
    print(f"gpu_name: {gpu_name or 'unavailable'}")
    print(f"mode: {mode}")
    print("Press Ctrl+C to stop. Partial transcripts are exploratory, not final utterances.")


def print_partial_event(event: dict) -> None:
    transcript = str(event.get("transcript", ""))
    latency_ms = float(event.get("latency_ms", 0.0))
    rtf = float(event.get("rtf", 0.0))
    message = f"\r{transcript}  latency_ms={latency_ms:.2f} rtf={rtf:.3f}"
    print(message, end="", flush=True)


def print_session_summary(summary: dict) -> None:
    print("Session summary")
    print(f"total_runtime_s: {_format_optional_float(summary['total_runtime_s'])}")
    print(f"inference_windows: {summary['inference_windows']}")
    print(f"avg_latency_ms: {_format_optional_float(summary['avg_latency_ms'])}")
    print(f"p50_latency_ms: {_format_optional_float(summary['p50_latency_ms'])}")
    print(f"p95_latency_ms: {_format_optional_float(summary['p95_latency_ms'])}")
    print(f"avg_rtf: {_format_optional_float(summary['avg_rtf'])}")
    print(f"dropped_audio_s: {_format_optional_float(summary['dropped_audio_s'])}")
    if summary["dropped_audio_s"] > 0:
        # latency_ms only times the model call, so falling behind shows up here instead.
        print(
            "warning: inference fell behind the microphone; the oldest audio was dropped and "
            "transcripts skip it. Try a longer --inference-window-ms or a faster device."
        )


def chunk_ms_to_samples(chunk_ms: int, sample_rate: int) -> int:
    if chunk_ms <= 0:
        raise ValueError("--chunk-ms must be positive")
    return max(1, int(sample_rate * (chunk_ms / 1000)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_demo(args)
    except (RuntimeError, ValueError, ImportError) as exc:
        print(f"Streaming demo failed: {exc}", file=sys.stderr)
        return 1


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
