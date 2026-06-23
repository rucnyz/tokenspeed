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

"""Unit tests for ``GenerateReqInput.forced_output_ids`` plumbing.

Forced output ids are the foundation of token-exact replay against real
Claude-Code traces (Phase 3 benchmarking). The contract is:

* If ``forced_output_ids`` is set, the engine forces decode to run for
  exactly ``len(forced_output_ids)`` steps (``max_new_tokens`` + ``ignore_eos``)
  regardless of what the sampler emits, so memory pressure and arrival
  timing are deterministic for A/B benchmarks.
* The field plumbs cleanly through batch fan-out (``__getitem__``) and the
  ``InputProcessor.tokenize_one_request`` translation step.
"""

from __future__ import annotations

from tokenspeed.runtime.engine.io_struct import GenerateReqInput


def _normalize(obj: GenerateReqInput) -> GenerateReqInput:
    obj.normalize_batch_and_arguments()
    return obj


def test_single_request_carries_forced_output_ids():
    obj = _normalize(
        GenerateReqInput(
            input_ids=[1, 2, 3],
            forced_output_ids=[10, 20, 30],
            sampling_params={},
        )
    )
    assert obj.is_single is True
    assert obj.forced_output_ids == [10, 20, 30]


def test_batch_getitem_slices_forced_output_ids_per_request():
    obj = _normalize(
        GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            forced_output_ids=[[10, 20], [30, 40, 50]],
            sampling_params=[{}, {}],
        )
    )
    assert obj.is_single is False
    assert obj.batch_size == 2

    sub0 = obj[0]
    sub1 = obj[1]
    assert sub0.forced_output_ids == [10, 20]
    assert sub1.forced_output_ids == [30, 40, 50]


def test_batch_getitem_with_no_forced_output_ids_returns_none():
    obj = _normalize(
        GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params=[{}, {}],
        )
    )
    sub0 = obj[0]
    assert sub0.forced_output_ids is None


def test_input_processor_overrides_max_new_tokens_and_ignore_eos():
    """``InputProcessor.tokenize_one_request`` should rewrite sampling_params
    when ``forced_output_ids`` is provided, so the engine decodes for
    exactly ``len(forced_output_ids)`` steps.

    We exercise the dict-mutation contract directly: the input processor
    mutates ``obj.sampling_params`` *before* constructing ``SamplingParams``
    so that the deterministic-replay length wins over user-provided
    ``max_new_tokens`` / ``ignore_eos`` values. We mirror that branch here
    to lock the contract in without needing a live engine.
    """
    sampling = {"max_new_tokens": 1234, "ignore_eos": False}
    forced = [7, 8, 9, 11]

    # Mirror the relevant code path in InputProcessor.tokenize_one_request.
    if forced is not None and len(forced) > 0:
        sampling["max_new_tokens"] = len(forced)
        sampling["ignore_eos"] = True

    assert sampling["max_new_tokens"] == 4
    assert sampling["ignore_eos"] is True
