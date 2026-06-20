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

    def execute_async(self, plan: FirePlan) -> None:
        worker = threading.Thread(target=self._execute_locked, args=(plan,), daemon=True)
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
