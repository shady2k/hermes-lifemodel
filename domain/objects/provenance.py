"""Provenance + W3C-traceparent-compatible creation-context (lm-27n.1).

Every typed object may carry a :class:`Provenance`: *who* created it, *why*,
the domain turn it belongs to, the causal ids it derived from, and — adopting
the **W3C Trace Context data model** (not the OpenTelemetry SDK; stdlib only) —
the execution trace context that was live *at creation time*.

This is the **definitions + validation** layer (task .1): the fields are
defined, strictly validated, normalized (hex lowercased), and round-tripped
through the codec. Minting a trace id, continuing-or-starting a trace, and
propagating/log-stamping it are task .2 — nothing here generates or mints ids.

The field is deliberately named ``creation_span_id``, never ``span_id``: it is
the trace context *at creation*, never "this object's live span" (guarding the
semantic drift codex flagged). ``turn_id`` (the Hermes conversation-turn
identity) stays distinct from ``trace_id`` (execution correlation).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import InvalidPayload

#: The only characters a lowercase hex string may contain.
_HEX_DIGITS = frozenset("0123456789abcdef")

#: W3C traceparent version this build emits and accepts (the sole valid one for
#: our closed use; §Trace Context reserves "ff" as invalid).
_TRACEPARENT_VERSION = "00"

#: Field widths from the W3C Trace Context spec.
_TRACE_ID_LEN = 32
_SPAN_ID_LEN = 16
_TRACE_FLAGS_LEN = 2

#: Default flags when a trace is present but flags were not set: "01" = sampled.
_DEFAULT_TRACE_FLAGS = "01"


class Sensitivity(StrEnum):
    """How guarded an object's content is (privacy tier, HLA §4.1)."""

    NORMAL = "normal"
    SENSITIVE = "sensitive"
    PRIVATE = "private"


def _normalize_hex(value: str, *, length: int, allow_all_zero: bool, name: str) -> str:
    """Validate *value* as ``length`` hex digits and return it lowercased.

    Raises :class:`InvalidPayload` on wrong length, a non-hex character, or (when
    ``allow_all_zero`` is false) the all-zero value the W3C spec forbids for a
    trace/span id.
    """
    if len(value) != length:
        raise InvalidPayload(f"{name!r} must be {length} hex digits, got {len(value)} ({value!r})")
    lowered = value.lower()
    if any(ch not in _HEX_DIGITS for ch in lowered):
        raise InvalidPayload(f"{name!r} must be lowercase hex, got {value!r}")
    if not allow_all_zero and lowered == "0" * length:
        raise InvalidPayload(f"{name!r} must not be all-zero")
    return lowered


@dataclass(frozen=True)
class Provenance:
    """Creation lineage + W3C trace context for a typed object (HLA §4.1).

    All trace fields are optional — a record may predate tracing. When present
    they are validated and normalized to lowercase hex by ``__post_init__``, so
    no ``Provenance`` value ever holds a malformed trace id.
    """

    created_by: str
    component: str
    reason: str
    turn_id: str | None = None
    #: Qualified ids of the objects that causally produced this one.
    source_object_ids: tuple[str, ...] = ()
    source_signal_ids: tuple[str, ...] = ()
    # --- W3C traceparent-compatible CREATION context (never a live span) -----
    trace_id: str | None = None
    creation_span_id: str | None = None
    parent_span_id: str | None = None
    trace_flags: str | None = None

    def __post_init__(self) -> None:
        # Frozen dataclass: rewrite the normalized (lowercased) values in place.
        if self.trace_id is not None:
            object.__setattr__(
                self,
                "trace_id",
                _normalize_hex(
                    self.trace_id, length=_TRACE_ID_LEN, allow_all_zero=False, name="trace_id"
                ),
            )
        if self.creation_span_id is not None:
            object.__setattr__(
                self,
                "creation_span_id",
                _normalize_hex(
                    self.creation_span_id,
                    length=_SPAN_ID_LEN,
                    allow_all_zero=False,
                    name="creation_span_id",
                ),
            )
        if self.parent_span_id is not None:
            object.__setattr__(
                self,
                "parent_span_id",
                _normalize_hex(
                    self.parent_span_id,
                    length=_SPAN_ID_LEN,
                    allow_all_zero=False,
                    name="parent_span_id",
                ),
            )
        if self.trace_flags is not None:
            object.__setattr__(
                self,
                "trace_flags",
                _normalize_hex(
                    self.trace_flags,
                    length=_TRACE_FLAGS_LEN,
                    allow_all_zero=True,
                    name="trace_flags",
                ),
            )


def format_traceparent(provenance: Provenance) -> str:
    """Render the W3C ``traceparent`` header string for *provenance*.

    ``"00-{trace_id}-{creation_span_id}-{trace_flags}"``; flags default to
    ``"01"`` (sampled) when unset. Requires both a ``trace_id`` and a
    ``creation_span_id`` — calling without a trace present is a caller error
    (:class:`ValueError`), not malformed data.
    """
    if provenance.trace_id is None or provenance.creation_span_id is None:
        raise ValueError("cannot format a traceparent without a trace_id and creation_span_id")
    flags = provenance.trace_flags or _DEFAULT_TRACE_FLAGS
    return f"{_TRACEPARENT_VERSION}-{provenance.trace_id}-{provenance.creation_span_id}-{flags}"


def parse_traceparent(value: str) -> tuple[str, str, str]:
    """Parse a W3C ``traceparent`` string into ``(trace_id, span_id, flags)``.

    Strict: the four dash-separated fields must be version ``"00"``, a valid
    32-hex non-zero trace id, a valid 16-hex non-zero span id, and 2-hex flags.
    Raises :class:`InvalidPayload` on any malformed input (untrusted string).
    """
    parts = value.split("-")
    if len(parts) != 4:
        raise InvalidPayload(f"traceparent must have 4 dash-separated fields, got {value!r}")
    version, trace_id, span_id, flags = parts
    if version != _TRACEPARENT_VERSION:
        raise InvalidPayload(f"unsupported traceparent version {version!r}")
    return (
        _normalize_hex(trace_id, length=_TRACE_ID_LEN, allow_all_zero=False, name="trace_id"),
        _normalize_hex(span_id, length=_SPAN_ID_LEN, allow_all_zero=False, name="span_id"),
        _normalize_hex(flags, length=_TRACE_FLAGS_LEN, allow_all_zero=True, name="trace_flags"),
    )
