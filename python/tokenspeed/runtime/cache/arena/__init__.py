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

"""CUDA VMM arena substrate for HiMA inter-pool capacity transfer."""

from __future__ import annotations

import os

ARENA_ENABLED = os.getenv("TOKENSPEED_ARENA", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

__all__ = [
    "ARENA_ENABLED",
    "ChunkArena",
    "SharedHandlePool",
    "from_blob",
    "XPoolActuator",
]


def __getattr__(name: str):
    if name == "ChunkArena":
        from tokenspeed.runtime.cache.arena.chunk_arena import ChunkArena

        return ChunkArena
    if name == "SharedHandlePool":
        from tokenspeed.runtime.cache.arena.shared_pool import SharedHandlePool

        return SharedHandlePool
    if name == "from_blob":
        from tokenspeed.runtime.cache.arena.from_blob import from_blob

        return from_blob
    if name == "XPoolActuator":
        from tokenspeed.runtime.cache.arena.xpool_actuator import XPoolActuator

        return XPoolActuator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
