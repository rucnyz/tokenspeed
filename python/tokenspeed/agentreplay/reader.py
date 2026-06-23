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

"""Trace file reader for ``traces/cc_qwen*.jsonl``.

Each line in a trace file represents one inference step issued by Claude
Code (real session, real tokenizer, real timing). The fields used here:

* ``t``                  -- wall-clock offset (seconds) from session start
* ``program_id``         -- session id; subagents have their own program_id
* ``parent_program_id``  -- the spawning session (None for the root)
* ``spawn_ts``           -- when the subagent was spawned (None for root)
* ``step``               -- 1-indexed step number within a (sub)session
* ``input_ids``          -- pre-tokenized full prompt for this step
* ``forced_output_ids``  -- target output for token-exact replay
* ``tool_gap_after``     -- think/tool time after the assistant response
                            before the next prompt in the same session
                            arrives (seconds)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayStep:
    """One row from a replay trace JSONL file.

    All fields are passed through verbatim from the dataset; consumers
    are responsible for honoring ``t`` / ``tool_gap_after`` for arrival
    timing and ``forced_output_ids`` for deterministic decode length.
    """

    t: float
    program_id: str
    step: int
    parent_program_id: str | None
    spawn_ts: float | None
    input_ids: list[int]
    forced_output_ids: list[int]
    tool_gap_after: float

    @property
    def is_subagent(self) -> bool:
        return self.parent_program_id is not None

    @property
    def prompt_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def output_tokens(self) -> int:
        return len(self.forced_output_ids)


def iter_trace(path: str) -> Iterator[ReplayStep]:
    """Yield :class:`ReplayStep` from a replay trace file."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # The legacy ``smoke.jsonl`` uses a different schema (``body`` /
            # ``output_len`` / ``ref_e2e_ms``) and lacks ``input_ids`` —
            # surface a clear error rather than silently skipping so the
            # caller knows to point at a token-exact trace instead.
            if "input_ids" not in obj or "forced_output_ids" not in obj:
                raise ValueError(
                    f"{path}: line missing input_ids/forced_output_ids; "
                    "expected the pre-tokenized cc_qwen*.jsonl format."
                )
            yield ReplayStep(
                t=float(obj["t"]),
                program_id=str(obj["program_id"]),
                step=int(obj.get("step", 0)),
                parent_program_id=obj.get("parent_program_id"),
                spawn_ts=(
                    float(obj["spawn_ts"]) if obj.get("spawn_ts") is not None else None
                ),
                input_ids=list(obj["input_ids"]),
                forced_output_ids=list(obj["forced_output_ids"]),
                # ``tool_gap_after`` is ``None`` on the final step of a
                # session (no successor to gap against); treat that as 0.
                tool_gap_after=float(obj.get("tool_gap_after") or 0.0),
            )


def sessions_in_order(path: str) -> list[tuple[str, list[ReplayStep]]]:
    """Group a trace into ``(program_id, steps)`` tuples.

    Steps within a session are sorted by ``step``. Sessions are returned
    in the order they first appear in the file (which is already the
    natural session-issuance order for the cc_qwen traces). Subagent
    sessions show up as their own top-level entries — their parent's
    spawn point is preserved via ``ReplayStep.spawn_ts`` and
    ``parent_program_id`` so the orchestrator can sequence them.
    """
    order: list[str] = []
    by_id: dict[str, list[ReplayStep]] = {}
    for step in iter_trace(path):
        if step.program_id not in by_id:
            order.append(step.program_id)
            by_id[step.program_id] = []
        by_id[step.program_id].append(step)
    return [(pid, sorted(by_id[pid], key=lambda s: s.step)) for pid in order]
