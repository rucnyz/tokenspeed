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

"""Shared physical handle pool for CUDA VMM arena."""

from __future__ import annotations

CHUNK_SIZE_BYTES = 2 * 1024 * 1024


class SharedHandlePool:
    """Logical pool of cuMemCreate handles keyed by chunk index."""

    def __init__(self, num_chunks: int) -> None:
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        self.num_chunks = num_chunks
        self.chunk_size = CHUNK_SIZE_BYTES
        self._handles: dict[int, int] = {}

    def get_handle(self, chunk_id: int) -> int:
        if chunk_id not in self._handles:
            self._handles[chunk_id] = chunk_id + 1
        return self._handles[chunk_id]
