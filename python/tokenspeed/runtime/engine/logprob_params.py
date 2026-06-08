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

"""Dedicated request-config structure for logprob return.

TokenSpeed keeps ``SamplingParams`` focused on *how to sample*; logprob return
config is about *what to report*, so it lives here. It exposes
``logprobs`` / ``prompt_logprobs`` / ``logprob_token_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_LOGPROB_TOKEN_IDS = 128


@dataclass
class LogprobParams:
    # top-N logprobs per OUTPUT (sampled) token; None = off, 0 = chosen only.
    # N>0 (top-N + chosen) and -1 (full vocab) are reserved but NOT yet
    # supported: the sampler/output path only materializes the sampled token's
    # logprob, so verify() rejects them rather than silently returning a single
    # token. Only None / 0 are currently honored.
    logprobs: int | None = None
    # top-N logprobs per PROMPT token; same value semantics as ``logprobs``.
    # SCOPED OUT for now: verify() rejects it (only the single-chunk pure-extend
    # prompt path is validated; chunked/mixed/prefix-cache paths are unsafe).
    prompt_logprobs: int | None = None
    # logprobs for specific token ids, per position. SCOPED OUT for now:
    # verify() rejects it (only feeds the prompt/extend path, scoped out above).
    logprob_token_ids: list[int] | None = None
    # detokenize tokens in returned logprobs (was ``return_text_in_logprobs``).
    return_text: bool = False

    def num_logprobs(self) -> int | None:
        """Per-output-token logprob count, or None if not requested.

        NOTE: ``0`` is a valid requested state (chosen token only), distinct
        from ``None`` (not requested). Callers MUST branch on ``is None``, not
        on truthiness of the return value.
        """
        if self.logprobs is not None:
            return self.logprobs
        if self.logprob_token_ids:
            return len(self.logprob_token_ids)
        return None

    def num_prompt_logprobs(self) -> int | None:
        """Per-prompt-token logprob count, or None if not requested."""
        return self.prompt_logprobs

    @property
    def requested(self) -> bool:
        return (
            self.logprobs is not None
            or self.prompt_logprobs is not None
            or bool(self.logprob_token_ids)
        )

    def verify(self, vocab_size: int, max_logprobs: int) -> None:
        # TODO(logprobs): re-enable prompt_logprobs / logprob_token_ids.
        # The prompt-logprob path is only correct for a single-chunk, pure-extend
        # prefill (start_len=0, no cross-chunk offset). Chunked prompts
        # (prompt_tokens > chunked_prefill_size), mixed extend+decode batches,
        # and prefix-cache hits currently fall outside the validated path and
        # would return silently wrong/partial results. Scope this PR to OUTPUT
        # logprobs only and reject the prompt surface loudly until the chunked
        # path lands. logprob_token_ids is rejected here too because it only
        # feeds the prompt/extend path (see logits_processor.get_token_ids_logprobs).
        if self.prompt_logprobs is not None:
            raise ValueError(
                "prompt_logprobs is not supported yet; only output logprobs "
                "are available."
            )
        if self.logprob_token_ids:
            raise ValueError(
                "logprob_token_ids is not supported yet; only output logprobs "
                "are available."
            )

        if self.logprobs is not None:
            if self.logprobs < -1:
                raise ValueError(f"logprobs must be >= -1, got {self.logprobs}.")
            if self.logprobs == -1:
                # Full-vocab logprobs are not implemented yet (the sampler/output
                # path does not materialize a vocab-sized result). Reject loudly
                # instead of silently returning a single-token logprob dict.
                raise ValueError(
                    "logprobs=-1 (full-vocab logprobs) is not supported yet; "
                    "use logprobs=0 (the sampled token's logprob)."
                )
            if self.logprobs > 0:
                # TODO(logprobs): re-enable output top-k. The sampler/output path
                # only emits the sampled token's logprob today, so logprobs>0
                # would silently drop the requested top-N alternatives. Reject
                # until output top-k is computed in the sampler and captured into
                # the CUDA-graph output buffers. Only logprobs=0 is honored.
                raise ValueError(
                    f"logprobs={self.logprobs} (output top-k) is not supported "
                    "yet; only logprobs=0 (the sampled token's logprob) is "
                    "available."
                )
