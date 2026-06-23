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

"""Token-exact replay against real Claude-Code traces.

This package drives an in-process :class:`tokenspeed.runtime.entrypoints.engine.Engine`
with prompts from the ``UCSB-SURFI/claude-code-traces`` dataset, preserving
the original arrival timing and per-step decode length. The motivation is
that scheduling/policy A/B experiments (HiMA Phase 3) need a workload that
looks like real agentic traffic, not synthetic random prompts.
"""

from tokenspeed.agentreplay.metrics import (
    PerRequestMetric,
    aggregate,
    write_jsonl,
)
from tokenspeed.agentreplay.reader import (
    ReplayStep,
    iter_trace,
    sessions_in_order,
)

__all__ = [
    "PerRequestMetric",
    "ReplayStep",
    "aggregate",
    "iter_trace",
    "sessions_in_order",
    "write_jsonl",
]
