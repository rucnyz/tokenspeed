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

"""Python-side XPool fire actuator executed on a background thread."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena


@dataclass(slots=True)
class FirePlan:
    direction: str
    page_ids: list[int]
    op_id: int = 0


class XPoolActuator:
    """Execute cuMemUnmap/Map transfers off the decode critical path."""

    def __init__(self, *, kv_arena: ChunkArena, mamba_arena: ChunkArena) -> None:
        self.kv_arena = kv_arena
        self.mamba_arena = mamba_arena
        self._lock = threading.Lock()
        self._inflight = False
        # The C++ budgeter latches its pending plan (op_id starts at 1), so we
        # de-dup here by the last actuated op_id; 0 means "nothing yet".
        self._last_op_id = 0

    def maybe_execute(self, plan: object) -> bool:
        """Actuate a budgeter plan unless it was already actuated.

        Args:
            plan: any object exposing ``op_id`` (int), ``direction`` (str) and
                ``page_ids`` (iterable of int) -- e.g. the C++ ``XPoolFirePlan``.

        Returns:
            True if a new transfer was launched, False if the plan was a
            duplicate of the previously actuated one.
        """
        op_id = int(plan.op_id)
        if op_id == self._last_op_id:
            return False
        self._last_op_id = op_id
        self.execute_async(
            FirePlan(
                direction=str(plan.direction),
                page_ids=list(plan.page_ids),
                op_id=op_id,
            )
        )
        return True

    def execute_async(self, plan: FirePlan) -> None:
        worker = threading.Thread(
            target=self._execute_locked, args=(plan,), daemon=True
        )
        worker.start()

    def _execute_locked(self, plan: FirePlan) -> None:
        with self._lock:
            if self._inflight:
                raise RuntimeError("XPoolActuator: concurrent fire is not supported")
            self._inflight = True
            try:
                if plan.direction == "mamba_to_kv":
                    self.mamba_arena.transfer_out(plan.page_ids)
                    self.kv_arena.transfer_in(plan.page_ids)
                elif plan.direction == "kv_to_mamba":
                    self.kv_arena.transfer_out(plan.page_ids)
                    self.mamba_arena.transfer_in(plan.page_ids)
                else:
                    raise ValueError(f"unknown fire direction: {plan.direction}")
            finally:
                self._inflight = False
