"""Machine-readable benchmark reports without collapsing dimensions."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    count: int
    accuracy: float
    nll: float | None = None
    brier: float | None = None
    ece: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None


def run_benchmark(
    name: str,
    rows: Sequence[Any],
    predict: Callable[[Any], Any],
    score: Callable[[Sequence[Any], Sequence[Any]], dict[str, float]],
) -> BenchmarkResult:
    outputs, latencies = [], []
    for row in rows:
        start = time.perf_counter()
        outputs.append(predict(row))
        latencies.append((time.perf_counter() - start) * 1000)
    metrics = score(rows, outputs)
    ordered = sorted(latencies)

    def percentile(q: float) -> float:
        return (
            ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))]
            if ordered
            else 0.0
        )

    return BenchmarkResult(
        name=name,
        count=len(rows),
        accuracy=float(metrics.get("accuracy", 0.0)),
        nll=metrics.get("nll"),
        brier=metrics.get("brier"),
        ece=metrics.get("ece"),
        latency_ms_p50=percentile(0.5),
        latency_ms_p95=percentile(0.95),
    )


def save_report(
    path: str | Path, results: Sequence[BenchmarkResult], *, metadata: dict[str, Any] | None = None
) -> None:
    payload = {
        "format_version": 1,
        "metadata": metadata or {},
        "results": [asdict(item) for item in results],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
