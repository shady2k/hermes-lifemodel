"""Pre-deploy smoke probe for the platform-adapter shell (bead lm-dte).

`make check` runs under `uv`, whose venv lacks Hermes' `gateway` package, so
`BasePlatformAdapter` is invisible off-host (mypy sees `Any`, pytest can't
instantiate the adapter). This probe closes that blind spot: run under the Hermes
venv, it imports `BeingAdapter` (an import/shell regression surfaces here) and
asserts it implements every abstract method of the *actually-installed* base — the
version-skew guard that catches the get_chat_info class of failure, where a missing
`@abstractmethod` only blows up at gateway connect time.

The load-bearing check needs no config. It deliberately does NOT construct the
adapter: `BeingAdapter.__init__` calls `Platform(PLATFORM_NAME)`, and the real
gateway only makes that platform name valid *after* `register()` runs
`ctx.register_platform(...)` — so faithful construction would require replicating
gateway-internal platform registration (fragile, drift-prone coupling). `run_smoke`
still accepts an optional `construct` thunk (exercised by unit tests, and available
if a future host makes standalone construction cheap). It never starts the brain
loop and never touches the live being.
"""

from __future__ import annotations

from collections.abc import Callable


class SmokeFailure(Exception):
    """A pre-deploy adapter-shell check failed."""


def run_smoke(adapter_cls: type, construct: Callable[[], object] | None = None) -> None:
    """Assert *adapter_cls* is fully concrete; if *construct* is given, that it succeeds.

    The abstract-method assertion is the load-bearing check (needs no gateway
    context). *construct* is optional because faithful standalone construction of
    the being adapter requires gateway-internal platform registration; the
    ``__main__`` entry omits it. Raises :class:`SmokeFailure` (never a bare
    AssertionError/arbitrary error) so the entry can write one clean message and
    exit non-zero.
    """
    missing: frozenset[str] = getattr(adapter_cls, "__abstractmethods__", frozenset())
    if missing:
        raise SmokeFailure(
            f"{adapter_cls.__name__} has unimplemented abstract methods: "
            f"{sorted(missing)} — the installed gateway base declares abstract methods "
            f"this adapter does not implement (it would fail to instantiate at connect)."
        )
    if construct is None:
        return
    try:
        construct()
    except Exception as exc:  # noqa: BLE001 - normalize every construction failure
        raise SmokeFailure(f"{adapter_cls.__name__} construction failed: {exc!r}") from exc


def _main() -> int:
    import sys

    try:
        # Import here (not at module top) so an import/shell regression is caught as a
        # smoke failure, and so `run_smoke`'s unit tests never import gateway.
        from .adapters.being_platform import BeingAdapter

        run_smoke(BeingAdapter)  # abstract-method (version-skew) guard; no construction
    # Direct stream writes, NOT print(): this is a standalone CLI entry (not a tick
    # component), and the trace-invariant guard bans print() across the whole runtime
    # tree so all being-runtime output goes through a SpanLogger. A dev/deploy CLI
    # legitimately writes its result to the terminal, so it uses the streams directly.
    except SmokeFailure as exc:
        sys.stderr.write(f"SMOKE FAIL: {exc}\n")
        return 1
    except Exception as exc:  # noqa: BLE001 - e.g. an import error is also a smoke failure
        sys.stderr.write(f"SMOKE FAIL: import/setup error: {exc!r}\n")
        return 1
    sys.stdout.write("SMOKE OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
