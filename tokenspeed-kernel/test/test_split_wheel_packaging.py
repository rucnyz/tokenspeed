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

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import patch

_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"


def _load_setup(monkeypatch, package: str):
    monkeypatch.chdir(_PYTHON_DIR)
    monkeypatch.setenv("TOKENSPEED_KERNEL_PACKAGE", package)
    monkeypatch.setenv("TOKENSPEED_KERNEL_VERSION_DATE", "20260101")
    monkeypatch.setenv("TOKENSPEED_KERNEL_GIT_SHA", "abcdef12")
    monkeypatch.setenv("TOKENSPEED_KERNEL_GIT_BRANCH", "test")

    with patch("setuptools.setup") as setup_mock:
        namespace = runpy.run_path(str(_PYTHON_DIR / "setup.py"), run_name="__main__")

    setup_mock.assert_called_once()
    return namespace, setup_mock.call_args.kwargs


def _requires_core(requirements: list[str]) -> bool:
    return any(req.startswith("tokenspeed-kernel==") for req in requirements)


def test_core_package_metadata_and_extras(monkeypatch):
    namespace, kwargs = _load_setup(monkeypatch, "core")

    assert kwargs["name"] == "tokenspeed-kernel"
    assert kwargs["entry_points"] == {}
    assert kwargs["extras_require"].keys() == {"nvidia", "amd", "all"}
    assert kwargs["extras_require"]["nvidia"] == [
        f"tokenspeed-kernel-nvidia[registration]=={kwargs['version']}"
    ]
    assert kwargs["extras_require"]["amd"] == [
        f"tokenspeed-kernel-amd[registration]=={kwargs['version']}"
    ]
    assert "apache-tvm-ffi>=0.1.5" in kwargs["install_requires"]
    assert "tokenspeed-deepgemm==2.5.0.post20260424" not in kwargs["install_requires"]
    assert "tokenspeed_kernel_nvidia" not in kwargs["packages"]
    assert "tokenspeed_kernel_amd" not in kwargs["packages"]

    include_module = namespace["_include_module"]
    assert include_module("tokenspeed_kernel.ops.attention.triton")
    assert not include_module("tokenspeed_kernel.ops.attention.flashinfer")
    assert not include_module("tokenspeed_kernel.ops.gemm.fp8_utils")
    assert not include_module("tokenspeed_kernel_nvidia.registration")


def test_nvidia_package_metadata_without_core_dependency(monkeypatch):
    namespace, kwargs = _load_setup(monkeypatch, "nvidia")

    assert kwargs["name"] == "tokenspeed-kernel-nvidia"
    assert not _requires_core(kwargs["install_requires"])
    assert kwargs["extras_require"] == {
        "registration": [f"tokenspeed-kernel=={kwargs['version']}"]
    }
    assert kwargs["entry_points"] == {
        "tokenspeed_kernel.plugins": ["nvidia=tokenspeed_kernel_nvidia:register"]
    }
    assert "tokenspeed_kernel_nvidia" in kwargs["packages"]
    assert "tokenspeed_kernel_amd" not in kwargs["packages"]

    include_module = namespace["_include_module"]
    assert include_module("tokenspeed_kernel_nvidia.registration")
    assert include_module("tokenspeed_kernel.ops.gemm.fp8_utils")
    assert include_module("tokenspeed_kernel.ops.moe.expert_location_dispatch")
    assert not include_module("tokenspeed_kernel.ops.gemm")
    assert not include_module("tokenspeed_kernel.ops.attention.gluon")


def test_amd_package_metadata_without_core_dependency(monkeypatch):
    namespace, kwargs = _load_setup(monkeypatch, "amd")

    assert kwargs["name"] == "tokenspeed-kernel-amd"
    assert not _requires_core(kwargs["install_requires"])
    assert kwargs["extras_require"] == {
        "registration": [f"tokenspeed-kernel=={kwargs['version']}"]
    }
    assert kwargs["entry_points"] == {
        "tokenspeed_kernel.plugins": ["amd=tokenspeed_kernel_amd:register"]
    }
    assert "tokenspeed_kernel_amd" in kwargs["packages"]
    assert "tokenspeed_kernel_nvidia" not in kwargs["packages"]

    include_module = namespace["_include_module"]
    assert include_module("tokenspeed_kernel_amd.registration")
    assert include_module("tokenspeed_kernel.ops.attention.gluon")
    assert include_module("tokenspeed_kernel.ops.moe.expert_location_dispatch")
    assert not include_module("tokenspeed_kernel.ops.moe")
    assert not include_module("tokenspeed_kernel.ops.gemm.deep_gemm")
