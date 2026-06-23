# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Per-request and aggregate metrics for replay runs.

We keep the schema deliberately small so the same JSONL/JSON files can be
ingested by ``compare_runs.py`` (table) and any future plotting code
without a separate parser layer.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass
class PerRequestMetric:
    """One emitted record per replayed inference step.

    All timestamps are in seconds, monotonic from the harness start
    (NOT wall-clock); ``-1`` indicates the event did not occur.
    """

    rid: str
    program_id: str
    step: int
    parent_program_id: str | None
    arrival_t: float
    first_token_t: float
    finish_t: float
    prompt_tokens: int
    output_tokens: int
    cached_tokens: int
    finish_reason: str
    # If non-empty, an explanation of how the request failed (timeout,
    # abort, etc.) -- the harness keeps going so a single bad request
    # does not kill the whole replay.
    error: str = ""

    @property
    def ttft_ms(self) -> float:
        if self.first_token_t < 0 or self.arrival_t < 0:
            return float("nan")
        return (self.first_token_t - self.arrival_t) * 1000.0

    @property
    def latency_ms(self) -> float:
        if self.finish_t < 0 or self.arrival_t < 0:
            return float("nan")
        return (self.finish_t - self.arrival_t) * 1000.0


def write_jsonl(records: Iterable[PerRequestMetric], path: str) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec)) + "\n")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(v for v in values if not math.isnan(v))
    if not sorted_vals:
        return float("nan")
    k = max(
        0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1))))
    )
    return sorted_vals[k]


def aggregate(
    records: list[PerRequestMetric],
    *,
    wall_start: float,
    wall_end: float,
) -> dict:
    """Compute summary metrics suitable for paper-style A/B tables.

    Includes TTFT p50/p99, end-to-end latency p50/p99, throughput in
    output tokens / second (over wall time), prefix-cache hit ratio,
    and bookkeeping counters. We surface ``failed`` separately rather
    than rolling failures into the latency stats -- a failed request
    has no meaningful TTFT to mix in.
    """
    finished = [r for r in records if not r.error and r.finish_t > 0]
    failed = [r for r in records if r.error]

    ttfts = [r.ttft_ms for r in finished if not math.isnan(r.ttft_ms)]
    latencies = [r.latency_ms for r in finished if not math.isnan(r.latency_ms)]

    total_output_tokens = sum(r.output_tokens for r in finished)
    total_prompt_tokens = sum(r.prompt_tokens for r in finished)
    total_cached = sum(r.cached_tokens for r in finished)

    wall = max(1e-9, wall_end - wall_start)

    return {
        "n_requests": len(records),
        "n_finished": len(finished),
        "n_failed": len(failed),
        "wall_seconds": wall,
        "ttft_ms_p50": _percentile(ttfts, 50),
        "ttft_ms_p90": _percentile(ttfts, 90),
        "ttft_ms_p99": _percentile(ttfts, 99),
        "ttft_ms_mean": statistics.mean(ttfts) if ttfts else float("nan"),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p99": _percentile(latencies, 99),
        "output_tokens_per_s": total_output_tokens / wall,
        "prompt_tokens_total": total_prompt_tokens,
        "output_tokens_total": total_output_tokens,
        "cached_tokens_total": total_cached,
        "cache_hit_ratio": (
            total_cached / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
        ),
    }
