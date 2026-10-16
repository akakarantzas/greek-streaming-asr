from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from src.profiling.stats import mean as _mean
from src.profiling.stats import percentile as _percentile


class LatencyTracker:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._active_starts: dict[str | None, float] = {}
        self._records: list[dict] = []

    def start(self, label: str | None = None) -> None:
        self._active_starts[label] = self._clock()

    def stop(self, label: str | None = None) -> float:
        if label not in self._active_starts:
            raise ValueError(f"No active latency timer for label {label!r}")
        start_time = self._active_starts.pop(label)
        latency_ms = (self._clock() - start_time) * 1000
        self.record(latency_ms=latency_ms, label=label)
        return latency_ms

    def record(self, latency_ms: float, label: str | None = None) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        self._records.append({"latency_ms": float(latency_ms), "label": label})

    def summary(self) -> dict:
        latencies = [record["latency_ms"] for record in self._records]
        return {
            "count": len(latencies),
            "min_latency_ms": min(latencies) if latencies else None,
            "max_latency_ms": max(latencies) if latencies else None,
            "mean_latency_ms": _mean(latencies),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p99_latency_ms": _percentile(latencies, 99),
        }

    def to_json(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.summary(), indent=2) + "\n",
            encoding="utf-8",
        )