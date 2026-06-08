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

"""Logprob detokenization for the async frontend.

A dedicated processor turns sampler-produced logprob arrays
(``recv_obj.{input,output}_{token,top}_logprobs_{val,idx}``) into the
``logprobs_info`` payload the per-request ``RequestOutputCollector``
merges. Lives next to ``OutputProcessor`` rather than inside it so
F.2's empty-logprob root-cause fix has a single, isolated home.

The engine reference lets us read the live ``tokenizer`` (mutated on
``update_weights_from_disk``) without snapshotting it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.engine.logprob_params import LogprobParams

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.engine.io_struct import BatchStrOut


@dataclass
class Logprob:
    """Per-token logprob entry.

    Attributes:
        logprob: log-probability of the token.
        rank: Slot rank — 0 for the sampled token, >=1 for top-N alternatives, None if not ranked (e.g. logprob_token_ids entries).
        decoded_token: detokenized string, or None when text not requested.
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


# One position maps token_id -> Logprob.
LogprobsOnePosition = dict[int, "Logprob"]


def build_position_logprobs(
    vals: list[float],
    ids: list[int],
    decoded: list[str] | None,
) -> dict[int, Logprob]:
    """Build one position's {token_id: Logprob} dict.

    Convention: slot 0 is the sampled/target token (reported with its
    slot-0 rank), slots 1..N are the top-N alternatives (ranks 1..N). Keying a
    dict by token_id makes dedup free: when a sampled token also appears in the
    top-N, the slot-0 entry (lower rank) is kept and the duplicate is dropped.

    # Ranks may be non-contiguous if the sampled token duplicates a top-k slot.
    """
    ranks = itertools.chain((0,), range(1, len(ids)))
    out: dict[int, Logprob] = {}
    for k, (tid, lp, rank) in enumerate(zip(ids, vals, ranks)):
        if tid in out:
            # First (lowest-rank) write wins; e.g. the sampled slot-0 token
            # shadows its own appearance among the top-N alternatives.
            continue
        tok = decoded[k] if decoded is not None else None
        out[tid] = Logprob(logprob=float(lp), rank=int(rank), decoded_token=tok)
    return out


class LogprobsProcessor:
    """Translate sampler logprob arrays into per-request meta_info entries.

    Holds an engine reference solely for the live ``tokenizer`` it needs when
    the caller requests text decoding (``LogprobParams.return_text=True``).
    When ``return_text`` is False the tokenizer is never touched, which is the
    mode the stub-tokenizer test paths exercise.
    """

    def __init__(self, engine: AsyncLLM) -> None:
        self.engine = engine

    def convert_logprob_style(
        self,
        logprobs_info: dict,
        logprob_params: LogprobParams,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ) -> None:
        """Emit logprob dicts into ``logprobs_info``.

        Produces ``logprobs_info["logprobs"]`` and/or
        ``logprobs_info["prompt_logprobs"]`` as ``list[dict[int, Logprob]]``
        (one dict per token position) plus a running ``cumulative_logprob``
        (sum of the per-position slot-0/sampled logprobs). Lists EXTEND across
        streamed frames, so this may be called repeatedly for one request.
        """

        # Defensive: sampler may not have populated logprobs for this request
        # (e.g. backend doesn't support logprobs, overlap race). Treat missing
        # or out-of-range wire fields as empty rather than crashing the loop.
        def _get(field: str):
            lst = getattr(recv_obj, field, None) or []
            if recv_obj_index < len(lst):
                return lst[recv_obj_index]
            return []

        num_out = logprob_params.num_logprobs()
        num_prompt = logprob_params.num_prompt_logprobs()
        want_token_ids = bool(logprob_params.logprob_token_ids)

        if num_out is not None:
            out_sampled_val = _get("output_token_logprobs_val")
            positions = self._build_positions(
                sampled_val=out_sampled_val,
                sampled_idx=_get("output_token_logprobs_idx"),
                top_val=_get("output_top_logprobs_val"),
                top_idx=_get("output_top_logprobs_idx"),
                tid_val=_get("output_token_ids_logprobs_val") if want_token_ids else [],
                tid_idx=_get("output_token_ids_logprobs_idx") if want_token_ids else [],
                return_text=logprob_params.return_text,
            )
            logprobs_info.setdefault("logprobs", []).extend(positions)
            logprobs_info["cumulative_logprob"] = logprobs_info.get(
                "cumulative_logprob", 0.0
            ) + self._sum_slot0(out_sampled_val)

        if num_prompt is not None:
            in_sampled_val = _get("input_token_logprobs_val")
            positions = self._build_positions(
                sampled_val=in_sampled_val,
                sampled_idx=_get("input_token_logprobs_idx"),
                top_val=_get("input_top_logprobs_val"),
                top_idx=_get("input_top_logprobs_idx"),
                tid_val=_get("input_token_ids_logprobs_val") if want_token_ids else [],
                tid_idx=_get("input_token_ids_logprobs_idx") if want_token_ids else [],
                return_text=logprob_params.return_text,
            )
            # The engine emits, per prompt position k, the logprob of token
            # ``prompt_ids[k+1]`` (next-token), with a trailing sampled-position
            # entry. Convert to the output convention: ``prompt_logprobs[j]`` is the
            # logprob of ``prompt_ids[j]``, and ``prompt_logprobs[0]`` is None
            # (the first prompt token has no predecessor). So prepend None and
            # drop the trailing entry.
            if positions:
                positions = [None] + positions[:-1]
            logprobs_info.setdefault("prompt_logprobs", []).extend(positions)

    def _build_positions(
        self,
        sampled_val: list[float],
        sampled_idx: list[int],
        top_val: list[list[float]],
        top_idx: list[list[int]],
        tid_val: list[list[float]],
        tid_idx: list[list[int]],
        return_text: bool,
    ) -> list[dict[int, Logprob]]:
        """Assemble one ``dict[int, Logprob]`` per token position.

        For position ``p`` slot 0 is the sampled/target token
        (``sampled_*[p]``) and slots 1..N are the top-N alternatives
        (``top_*[p]``, possibly empty). When requested, the specific
        ``logprob_token_ids`` entries (``tid_*[p]``) are folded in without
        clobbering already-present (ranked) entries.
        """
        out: list[dict[int, Logprob]] = []
        for p in range(len(sampled_idx)):
            tvals = top_val[p] if p < len(top_val) and top_val[p] else []
            tids = top_idx[p] if p < len(top_idx) and top_idx[p] else []
            vals = [sampled_val[p]] + list(tvals)
            ids = [sampled_idx[p]] + list(tids)
            decoded = self._maybe_decode(ids, return_text)
            pos = build_position_logprobs(vals, ids, decoded)

            # Fold requested specific token-id logprobs into the position.
            xvals = tid_val[p] if p < len(tid_val) and tid_val[p] else []
            xids = tid_idx[p] if p < len(tid_idx) and tid_idx[p] else []
            if xids:
                xdec = self._maybe_decode(xids, return_text)
                for k, (xtid, xlp) in enumerate(zip(xids, xvals)):
                    if xtid in pos:
                        continue
                    # rank=None: the sampler wire does not emit a vocab rank for
                    # requested token-id logprobs; leaving it unranked is the
                    # only correct choice until the wire is extended.
                    pos[xtid] = Logprob(
                        logprob=float(xlp),
                        rank=None,
                        decoded_token=xdec[k] if xdec is not None else None,
                    )
            out.append(pos)
        return out

    def _maybe_decode(self, ids: list[int], return_text: bool) -> list[str] | None:
        if not return_text:
            return None
        assert self.engine.tokenizer is not None
        return self.engine.tokenizer.batch_decode(ids)

    @staticmethod
    def _sum_slot0(sampled_val: list[float]) -> float:
        return float(sum(sampled_val)) if sampled_val else 0.0
