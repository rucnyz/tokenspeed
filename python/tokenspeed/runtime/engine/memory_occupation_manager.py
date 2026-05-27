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

"""CPU-staged release/resume for torch_memory_saver-managed regions.

`torch_memory_saver.pause()` releases the physical GPU pages backing tensors
allocated in `saver.region()` and preserves their virtual addresses. It does
**not** copy data to host RAM, so the next `resume()` returns memory with
zeroed contents — callers either accept that (scratch buffers, KV cache) or
reload from disk (weights).

For the RLHF train↔serve handoff case it's cheaper to stage weights to host
RAM on release and copy them back on resume than to re-read tens-of-GiB
checkpoints from disk on every cycle. This module implements that pattern:

  - On `release(stage_to_cpu=True)`:
      1. Copy every model parameter into a pre-allocated pinned host buffer.
      2. Call `saver.pause()` to release the GPU pages.
  - On `resume()`:
      1. Call `saver.resume()` so the same virtual addresses are backed again.
      2. Copy each parameter back from the pinned host buffer.

Trade-off: host RAM holds a full duplicate of the model weights for the
duration of the release. For Qwen2-72B in bf16 that's ~145 GiB of host RAM
during the offload window. The release side becomes the bottleneck on the
HtoD path (PCIe Gen4 x16 ≈ 25 GiB/s, NVLink-attached host even faster).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

logger = logging.getLogger(__name__)


class MemoryOccupationManager:
    """Coordinates torch_memory_saver pause/resume with optional CPU staging."""

    def __init__(
        self,
        memory_saver: TorchMemorySaverAdapter,
        model_runner: ModelRunner | None,
        draft_model_runner: ModelRunner | None = None,
        use_pinned_host: bool = True,
    ) -> None:
        self.memory_saver = memory_saver
        self.model_runner = model_runner
        self.draft_model_runner = draft_model_runner
        self.use_pinned_host = use_pinned_host
        # name -> CPU tensor snapshot. Populated on release(stage_to_cpu=True),
        # consumed and cleared on resume().
        self._staged: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # release
    # ------------------------------------------------------------------

    def release(
        self, *, stage_to_cpu: bool = False, tags: list[str] | None = None
    ) -> None:
        """Release GPU memory; optionally stage model weights to host RAM first."""
        if stage_to_cpu:
            self._stage_to_cpu()
        self.memory_saver.pause()
        # Surrender allocator-cached and IPC pages to the driver as well.
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def resume(self, *, tags: list[str] | None = None) -> None:
        """Reacquire GPU memory; restore staged weights if any are pending."""
        self.memory_saver.resume()
        if self._staged:
            self._restore_from_cpu()

    # ------------------------------------------------------------------
    # internal: CPU staging
    # ------------------------------------------------------------------

    def _iter_runners(self):
        for runner in (self.model_runner, self.draft_model_runner):
            if runner is not None and getattr(runner, "model", None) is not None:
                yield runner

    def _stage_to_cpu(self) -> None:
        """Copy every model parameter into a pinned host buffer.

        Reuses existing snapshot buffers when present (same shape/dtype) so a
        repeated release→resume→release cycle doesn't reallocate host RAM.
        """
        for runner in self._iter_runners():
            model = runner.model
            prefix = "model" if runner is self.model_runner else "draft"
            for name, param in model.named_parameters():
                key = f"{prefix}::{name}"
                src = param.detach()
                snap = self._staged.get(key)
                if snap is None or snap.shape != src.shape or snap.dtype != src.dtype:
                    snap = torch.empty(
                        src.shape,
                        dtype=src.dtype,
                        device="cpu",
                        pin_memory=self.use_pinned_host and torch.cuda.is_available(),
                    )
                    self._staged[key] = snap
                snap.copy_(src, non_blocking=True)
            # Also stage buffers (norm running stats, embeddings stored as
            # buffers, etc.) so resume reconstructs the full forward state.
            for name, buf in model.named_buffers():
                key = f"{prefix}::buffer::{name}"
                src = buf.detach()
                snap = self._staged.get(key)
                if snap is None or snap.shape != src.shape or snap.dtype != src.dtype:
                    snap = torch.empty(
                        src.shape,
                        dtype=src.dtype,
                        device="cpu",
                        pin_memory=self.use_pinned_host and torch.cuda.is_available(),
                    )
                    self._staged[key] = snap
                snap.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        logger.info("Staged %d tensors to host RAM ahead of pause()", len(self._staged))

    def _restore_from_cpu(self) -> None:
        """Copy staged host-side tensors back into the original GPU buffers.

        The GPU buffers are at the same virtual addresses as before pause();
        their contents are zeroed at this point. ``param.data.copy_()`` writes
        through to those same pages, preserving every captured CUDAGraph
        argument pointer.
        """
        with torch.no_grad():
            for runner in self._iter_runners():
                model = runner.model
                prefix = "model" if runner is self.model_runner else "draft"
                for name, param in model.named_parameters():
                    key = f"{prefix}::{name}"
                    snap = self._staged.get(key)
                    if snap is not None:
                        param.data.copy_(snap, non_blocking=True)
                for name, buf in model.named_buffers():
                    key = f"{prefix}::buffer::{name}"
                    snap = self._staged.get(key)
                    if snap is not None:
                        buf.data.copy_(snap, non_blocking=True)
        torch.cuda.synchronize()
        logger.info(
            "Restored %d tensors from host RAM after resume()", len(self._staged)
        )
        # Drop the snapshots so subsequent resumes without a staging release
        # don't accidentally clobber freshly loaded weights.
        self._staged.clear()
