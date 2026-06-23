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

"""Async replay driver.

Spawns one asyncio task per session (and per subagent). Each task walks
its session's steps in order; between adjacent steps it sleeps for
``tool_gap_after / time_scale`` to model the original session's
inter-arrival delay. Subagent sessions are launched independently and
honor their ``spawn_ts`` offset.

All requests share a single :class:`Engine`; concurrency naturally
floats up and down as the original Claude-Code traffic does.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

# Engine is only needed at runtime; avoid importing at module level so that
# lightweight unit tests (which don't need a GPU) can import replayer without
# triggering the full CUDA initialisation chain.
from typing import TYPE_CHECKING, Any

from tokenspeed.agentreplay.metrics import PerRequestMetric
from tokenspeed.agentreplay.reader import ReplayStep, sessions_in_order

if TYPE_CHECKING:
    from tokenspeed.runtime.entrypoints.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    """Parameters that shape a single replay run.

    ``time_scale > 1`` compresses inter-arrival gaps -- e.g.
    ``time_scale=6`` makes a one-hour trace replay in ten minutes,
    increasing concurrency to stress the scheduler. ``max_sessions``
    truncates the trace to the first N sessions so smoke runs stay
    short. ``warmup_seconds`` skips the initial wall-clock window so
    we don't measure cold-start prefill into the metric set.

    The cc_qwen* traces store ``t`` as *absolute wall-clock* timestamps
    spanning weeks of real Claude-Code usage (median inter-session gap
    is hours). Two knobs reshape that into a benchmark-friendly axis:

    * ``normalize_time`` (default ``True``) shifts the first session's
      arrival to ``t = 0`` so we don't sleep for days before issuing
      the first prompt.
    * ``max_inter_session_gap_s`` (default ``5.0``) caps the gap between
      consecutive sessions' first-step arrivals. This collapses the
      idle gaps between real human typing sessions into something the
      benchmark can actually exercise. Set to ``None`` to disable.
    * Intra-session gaps (``tool_gap_after``) are never capped -- the
      think/tool delay inside one agent turn is part of the workload
      we want to measure.
    """

    trace_path: str
    time_scale: float = 1.0
    max_sessions: int | None = None
    warmup_seconds: float = 0.0
    request_timeout_s: float = 600.0
    normalize_time: bool = True
    max_inter_session_gap_s: float | None = 5.0
    # S3 replayer fix: skip steps whose prompt exceeds the model's context
    # window so we don't get ValueError from the engine.  None = no limit.
    max_prompt_tokens: int | None = None


async def _run_one_step(
    engine: "Any",
    step: ReplayStep,
    *,
    harness_start: float,
    metrics: list[PerRequestMetric],
    timeout_s: float,
    max_prompt_tokens: int | None = None,
) -> None:
    """Issue one request to the engine and record its metrics.

    Uses ``stream=True`` so the first chunk timestamp gives a true TTFT
    measurement (otherwise we only see the final response and conflate
    queueing + prefill + full decode into a single number).
    """
    arrival_t = time.monotonic() - harness_start
    first_token_t = -1.0
    finish_t = -1.0
    cached_tokens = 0
    completion_tokens = 0
    finish_reason = ""
    error = ""

    # S3 replayer fix: skip steps that exceed the configured context limit.
    # Emitting to the engine would produce a ValueError that terminates the
    # whole session task; instead we record a structured error and return so
    # the session can continue with subsequent steps.
    n_prompt = len(step.input_ids) if step.input_ids else step.prompt_tokens
    if max_prompt_tokens is not None and n_prompt > max_prompt_tokens:
        error = f"prompt_too_long_{n_prompt}"
        logger.warning(
            "replay rid=%s step=%d skipped: prompt_tokens=%d > max_prompt_tokens=%d",
            step.program_id,
            step.step,
            n_prompt,
            max_prompt_tokens,
        )
        metrics.append(
            PerRequestMetric(
                rid=f"{step.program_id}_step{step.step}",
                program_id=step.program_id,
                step=step.step,
                parent_program_id=step.parent_program_id,
                arrival_t=arrival_t,
                first_token_t=-1.0,
                finish_t=-1.0,
                prompt_tokens=n_prompt,
                output_tokens=0,
                cached_tokens=0,
                finish_reason="",
                error=error,
            )
        )
        return

    try:
        # Pin sampling temperature to 0 + ignore_eos so the engine reliably
        # decodes for exactly len(forced_output_ids) steps regardless of
        # what the underlying model would have sampled. The forced_output_ids
        # field handles the max_new_tokens override automatically.
        generator = await engine.async_generate(
            input_ids=step.input_ids,
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            forced_output_ids=step.forced_output_ids,
            stream=True,
        )

        async def _consume():
            nonlocal first_token_t, finish_t, cached_tokens, completion_tokens, finish_reason
            async for chunk in generator:
                if first_token_t < 0:
                    first_token_t = time.monotonic() - harness_start
                # `meta_info` is updated in-place on the same dict each
                # iteration, so we read its final state after the loop
                # exits. We still need to touch the chunk here to drain
                # the async generator.
                meta = chunk.get("meta_info") or {}
                cached_tokens = int(meta.get("cached_tokens") or 0)
                completion_tokens = int(meta.get("completion_tokens") or 0)
                fr = meta.get("finish_reason")
                if fr:
                    finish_reason = fr if isinstance(fr, str) else fr.get("type", "")

        await asyncio.wait_for(_consume(), timeout=timeout_s)
        finish_t = time.monotonic() - harness_start
    except asyncio.TimeoutError:
        error = f"timeout_after_{timeout_s}s"
        logger.warning(
            "replay rid=%s step=%d timed out after %.1fs",
            step.program_id,
            step.step,
            timeout_s,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("replay rid=%s step=%d failed", step.program_id, step.step)

    metrics.append(
        PerRequestMetric(
            rid=f"{step.program_id}_step{step.step}",
            program_id=step.program_id,
            step=step.step,
            parent_program_id=step.parent_program_id,
            arrival_t=arrival_t,
            first_token_t=first_token_t,
            finish_t=finish_t,
            prompt_tokens=step.prompt_tokens,
            output_tokens=completion_tokens or step.output_tokens,
            cached_tokens=cached_tokens,
            finish_reason=finish_reason,
            error=error,
        )
    )


async def _replay_session(
    engine: "Any",
    steps: list[ReplayStep],
    *,
    harness_start: float,
    cfg: ReplayConfig,
    metrics: list[PerRequestMetric],
    arrive_at: float,
) -> None:
    """Replay one session's steps in order, honoring per-step delays.

    ``arrive_at`` is the *post-normalization* wall-clock offset at which
    the session's first step should be dispatched. The caller computes
    it from the trace; we just sleep until then.

    Within a session, step N+1 cannot start until step N has finished
    *and* its ``tool_gap_after`` (think + tool execution time) has
    elapsed. This mirrors how Claude Code waits for one turn before
    issuing the next.
    """
    if not steps:
        return
    delay = max(0.0, arrive_at - (time.monotonic() - harness_start))
    if delay > 0:
        await asyncio.sleep(delay)

    for i, step in enumerate(steps):
        await _run_one_step(
            engine,
            step,
            harness_start=harness_start,
            metrics=metrics,
            timeout_s=cfg.request_timeout_s,
            max_prompt_tokens=cfg.max_prompt_tokens,
        )
        if i + 1 < len(steps):
            gap = step.tool_gap_after / cfg.time_scale
            if gap > 0:
                await asyncio.sleep(gap)


def _compute_session_dispatch_times(
    sessions: list[tuple[str, list[ReplayStep]]],
    cfg: ReplayConfig,
) -> list[float]:
    """Map each session to the post-normalization arrival time (seconds).

    Steps:
      1. Pick a per-session "raw arrival" -- ``spawn_ts`` for subagents
         (so the parent's spawn point is honored) else ``t`` of the
         session's first step.
      2. Sort sessions by raw arrival (stable) so the gap-cap math is
         well-defined regardless of how the trace file orders them.
      3. Optionally normalize: subtract the minimum raw arrival.
      4. Optionally cap each gap-from-previous to
         ``max_inter_session_gap_s``. This collapses the multi-hour
         idle stretches between real human sessions into something a
         benchmark can exercise without changing intra-session timing.
      5. Apply ``time_scale``.

    Note we keep results aligned with the *input* ``sessions`` ordering
    (so the caller can zip them) -- only the sort during step 2 is
    internal.
    """
    n = len(sessions)
    if n == 0:
        return []

    raw_arrivals: list[tuple[int, float]] = []
    for idx, (_pid, steps) in enumerate(sessions):
        head = steps[0]
        raw = head.spawn_ts if head.spawn_ts is not None else head.t
        raw_arrivals.append((idx, float(raw)))

    raw_arrivals.sort(key=lambda x: x[1])
    sorted_raw = [t for _, t in raw_arrivals]
    base = sorted_raw[0] if cfg.normalize_time else 0.0
    cap = cfg.max_inter_session_gap_s

    dispatch_sorted: list[float] = []
    for i, raw in enumerate(sorted_raw):
        if i == 0:
            dispatch_sorted.append(max(0.0, raw - base))
            continue
        gap = raw - sorted_raw[i - 1]
        if cap is not None and gap > cap:
            gap = cap
        dispatch_sorted.append(dispatch_sorted[-1] + gap)

    out = [0.0] * n
    for (orig_idx, _), val in zip(raw_arrivals, dispatch_sorted):
        out[orig_idx] = val / cfg.time_scale
    return out


async def run_replay(
    engine: "Any", cfg: ReplayConfig
) -> tuple[list[PerRequestMetric], float, float]:
    """Drive a full replay against ``engine`` and collect per-request metrics.

    Returns ``(metrics, wall_start, wall_end)`` where the wall times
    bracket the post-warmup portion of the run (used by
    :func:`tokenspeed.agentreplay.metrics.aggregate` for throughput).
    """
    sessions = sessions_in_order(cfg.trace_path)
    if cfg.max_sessions is not None:
        sessions = sessions[: cfg.max_sessions]
    if not sessions:
        raise ValueError(f"No sessions found in {cfg.trace_path}")

    dispatch_times = _compute_session_dispatch_times(sessions, cfg)
    horizon = max(dispatch_times) if dispatch_times else 0.0
    logger.info(
        "replay starting: trace=%s sessions=%d time_scale=%.2f normalize=%s "
        "max_inter_session_gap=%s last_dispatch=%.1fs",
        cfg.trace_path,
        len(sessions),
        cfg.time_scale,
        cfg.normalize_time,
        cfg.max_inter_session_gap_s,
        horizon,
    )

    metrics: list[PerRequestMetric] = []
    harness_start = time.monotonic()
    wall_start = harness_start + cfg.warmup_seconds

    tasks = [
        asyncio.create_task(
            _replay_session(
                engine,
                steps,
                harness_start=harness_start,
                cfg=cfg,
                metrics=metrics,
                arrive_at=dispatch_times[i],
            )
        )
        for i, (_, steps) in enumerate(sessions)
    ]
    await asyncio.gather(*tasks, return_exceptions=False)

    wall_end = time.monotonic()
    return metrics, wall_start, wall_end
