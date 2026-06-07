"""Output-quality auditors.

Where the monitors answer "is the server *up* and *fast*", the auditors answer
"is the server's *output* actually correct". Each auditor is a pure function
``(ResponseRecord) -> list[Finding]`` registered by name, mirroring the
``workloads/`` registry. The client builds a ``ResponseRecord`` for every
completed response and runs the enabled auditors; findings are emitted as
``audit_finding`` events and aggregated post-run.

Keeping auditors pure (no I/O, no network) makes them trivially unit-testable
and composable, and cleanly separates per-response content checks from the
server-wide, time-windowed checks that live in ``monitors/`` (e.g. spec-decode
acceptance, which is only available as a global Prometheus counter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Severities, in increasing order of concern, and what they do to the run:
#   info  — diagnostic only.
#   warn  — recorded + shown in the report; heuristic or single-request scope
#           (e.g. one stalled request the engine may have lost track of).
#   error — recorded + shown; a response almost certainly wrong (tokens billed
#           but no content); gate-eligible via --fail-on-audit.
#   fatal — server-wide failure (e.g. a global decode wedge): the run is cut
#           off immediately, a report is still produced, and the exit is
#           nonzero regardless of any gate.
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_ERROR = "error"
SEVERITY_FATAL = "fatal"


@dataclass
class ResponseRecord:
    """Everything the client observed about one completed response.

    ``reported_*`` come from the server's ``usage`` block; ``observed_*`` are
    what the client actually saw on the wire. Keeping them separate is the
    whole point of the length auditor — they diverge when the engine bills
    completion tokens that carry no visible content (special tokens / empty
    deltas).
    """

    rid: str
    workload: str
    stream: bool
    content: str  # visible text (may be truncated to a cap by the client)
    content_truncated: bool
    reported_completion_tokens: int
    reported_prompt_tokens: int
    observed_visible_tokens: int  # stream: non-empty content deltas; else word count
    finish_reason: Optional[str]
    ttft_s: Optional[float]
    e2e_s: float
    max_inter_token_gap_s: float
    validate_schema: Optional[Dict] = None


@dataclass
class Finding:
    """One audit result. ``value`` carries the numeric the check keyed on, so
    the opt-in ``--fail-on-audit`` gate can threshold on it later."""

    check: str
    severity: str
    detail: str
    value: Optional[float] = None


@dataclass
class AuditConfig:
    """How the client should audit responses. Threaded through the runner."""

    enabled: tuple = ()  # auditor names to run; empty => none
    content_cap: int = 16384  # max chars of content to retain for auditing
    # Per-request inter-token gap that trips a (warn-level) request_stall; 0=off.
    stall_timeout_s: float = 20.0
    # Server-wide decode wedge: if requests are in decode (have produced a first
    # token, not finished) yet NObody yields another token for this long, the
    # engine has hung => fatal abort. Gated on in-decode requests so the window
    # can be short and still safe. 0 disables.
    global_stall_timeout_s: float = 20.0


Auditor = Callable[[ResponseRecord], List[Finding]]
_REGISTRY: Dict[str, Auditor] = {}


def register(name: str) -> Callable[[Auditor], Auditor]:
    def deco(fn: Auditor) -> Auditor:
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Auditor:
    if name not in _REGISTRY:
        raise KeyError(f"unknown auditor: {name}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> List[str]:
    return sorted(_REGISTRY)


def run_audits(record: ResponseRecord, enabled) -> List[Finding]:
    """Run the named auditors over ``record``; collect every finding.

    A broken auditor must never sink the request, so individual auditor
    exceptions are swallowed into an ``info`` finding rather than raised.
    """
    findings: List[Finding] = []
    for name in enabled:
        fn = _REGISTRY.get(name)
        if fn is None:
            continue
        try:
            findings.extend(fn(record))
        except Exception as e:  # noqa: BLE001 — an auditor bug must not fail the run
            findings.append(
                Finding(
                    check=name,
                    severity=SEVERITY_INFO,
                    detail=f"auditor raised {type(e).__name__}: {e}"[:200],
                )
            )
    return findings


# Import side-effects register the built-in auditors.
from . import builtin  # noqa: F401,E402
