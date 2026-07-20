"""Regression tests for PortArgs port-cluster isolation.

Guards against the bug where ``PortArgs.init_new`` picked ``nccl_port`` from a
*wide* random offset (``server_args.port + random.randint(100, 1000)``) that was
decoupled from the fixed ZMQ cluster. When several engines ran on one host only
~100 ports apart (e.g. parallel A/B eval arms on 30100/30200/30300), one
engine's random nccl port routinely landed inside a neighbour's ZMQ/gRPC port.
Combined with the non-atomic ``is_port_available`` TOCTOU check, the stray nccl
socket squatted a sibling's ZMQ endpoint, cross-talked pyobj traffic between
instances, and silently crashed the receiving main process under load.

The fix folds ``nccl_port`` into the same contiguous, collectively-scanned port
cluster so it is deterministic and disjoint. These tests pin that contract.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from tokenspeed.runtime.utils import server_args as sa_mod
from tokenspeed.runtime.utils.server_args import PortArgs


def _mk_server_args(port: int) -> SimpleNamespace:
    """Minimal stand-in exposing only the fields PortArgs.init_new reads."""
    mapping = SimpleNamespace(
        nnodes=1,
        has_attn_dp=False,
        attn=SimpleNamespace(dp_size=1),
    )
    return SimpleNamespace(port=port, dist_init_addr=None, mapping=mapping)


def _all_ports(port_args: PortArgs) -> set[int]:
    """Every host port a single engine binds/uses, extracted from PortArgs."""
    ports = {port_args.nccl_port}
    for name in (
        port_args.tokenizer_ipc_name,
        port_args.scheduler_input_ipc_name,
        port_args.rpc_ipc_name,
        port_args.metrics_ipc_name,
    ):
        ports.add(int(name.rsplit(":", 1)[-1]))
    return ports


class TestPortArgsIsolation(unittest.TestCase):
    def setUp(self):
        # Make the TOCTOU probe deterministic: every candidate looks free, so
        # init_new returns the first (base) cluster and we test pure derivation.
        patcher = mock.patch.object(sa_mod, "is_port_available", return_value=True)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_nccl_port_is_deterministic(self):
        """Same --port must yield the same nccl_port (no wide random offset)."""
        first = PortArgs.init_new(_mk_server_args(30100)).nccl_port
        for _ in range(20):
            self.assertEqual(PortArgs.init_new(_mk_server_args(30100)).nccl_port, first)

    def test_nccl_port_inside_cluster(self):
        """nccl_port sits at the tail of the contiguous ZMQ cluster."""
        pa = PortArgs.init_new(_mk_server_args(30100))
        dist_init_port = 30100 + sa_mod.ZMQ_TCP_PORT_DELTA
        # cluster: dist_init_port .. dist_init_port + 7 (nccl at the tail)
        self.assertEqual(pa.nccl_port, dist_init_port + 7)
        for p in _all_ports(pa):
            self.assertLessEqual(abs(p - dist_init_port), 10)

    def test_all_ports_unique_within_instance(self):
        pa = PortArgs.init_new(_mk_server_args(30100))
        ports = _all_ports(pa)
        # 5 distinct endpoints (tokenizer, scheduler_input, rpc, metrics, nccl).
        self.assertEqual(len(ports), 5)

    def test_parallel_arms_are_disjoint(self):
        """Core regression: arms spaced 100 apart must not share ANY port.

        Includes the historical nccl random window [port+100, port+1000], which
        used to overlap neighbouring arms.
        """
        arms = [30100, 30200, 30300]
        seen: dict[int, int] = {}
        for port in arms:
            pa = PortArgs.init_new(_mk_server_args(port))
            for p in _all_ports(pa):
                self.assertNotIn(
                    p,
                    seen,
                    msg=(
                        f"port {p} used by arm {port} collides with arm "
                        f"{seen.get(p)}"
                    ),
                )
                seen[p] = port


if __name__ == "__main__":
    unittest.main()
