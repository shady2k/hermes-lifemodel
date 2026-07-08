"""A tiny persisted JSON config for the lifemodel plugin (bead lm-j2w B2).

Distinct from :mod:`lifemodel.state.model` (the being's SOUL — vitals,
drives, timestamps, committed through the ``StatePort`` bus): this is plain
OWNER-facing settings that are not part of the cognition loop, starting with
the log level (``/lifemodel loglevel``). Lives next to ``lifemodel.sqlite`` in
the same per-profile ``base_dir`` (see :mod:`lifemodel.paths`), but as its own
small JSON file rather than a row in the state store — no schema versioning,
no tick-commit machinery needed for a handful of owner preferences.

stdlib-only (``json``, ``pathlib``, ``os``), matching the plugin's runtime
constraint (loaded inside Hermes' own interpreter, which may lack our dev
deps). :func:`read_config` is deliberately tolerant — a missing, empty, or
malformed file must never take the plugin down at load time; it just means
"no config yet", same as a fresh being. :func:`write_config` writes atomically
(temp file + ``os.replace``) so a crash mid-write can never leave a truncated
or partially-written config behind for the next read to choke on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .log import LOG_LEVEL_NAMES, configure, parse_log_level

#: The config file's name, sitting next to ``lifemodel.sqlite`` in the
#: per-profile ``base_dir``.
CONFIG_FILENAME = "config.json"

#: The config key the log level is stored under.
_LOG_LEVEL_KEY = "log_level"

#: The log level a being boots at when nothing has been persisted yet —
#: matches :func:`lifemodel.log.configure`'s own default.
DEFAULT_LOG_LEVEL = "info"


def read_config(base_dir: Path) -> dict[str, Any]:
    """Read the plugin's JSON config from *base_dir*.

    Tolerant by design: a missing file/directory, an empty file, or malformed
    JSON all return ``{}`` rather than raising — a config problem must never
    be load-bearing for the plugin coming up. A well-formed but non-object
    JSON document (e.g. a bare list) is likewise treated as "no config".
    """
    path = base_dir / CONFIG_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_config(base_dir: Path, config: dict[str, Any]) -> None:
    """Atomically persist *config* as JSON under *base_dir*.

    Writes to a sibling temp file then ``os.replace``s it over the real path
    — on POSIX (and modern Windows) ``replace`` is atomic, so a reader never
    observes a partially-written file, and a crash mid-write leaves the old
    config (or nothing) intact, never a corrupt one. Creates *base_dir* if it
    doesn't exist yet (mirrors :class:`lifemodel.events.EventSink`, which does
    the same for its sibling file in the same directory).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / CONFIG_FILENAME
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def read_log_level(base_dir: Path) -> str:
    """Return the persisted log level name, or :data:`DEFAULT_LOG_LEVEL`.

    Tolerant like :func:`read_config`: a missing/malformed config, or a
    ``log_level`` value that isn't a non-empty string, falls back to the
    default rather than raising — the command layer is where an owner-typed
    invalid name gets a validation error, not here.
    """
    level = read_config(base_dir).get(_LOG_LEVEL_KEY)
    if isinstance(level, str) and level.strip():
        return level.strip().lower()
    return DEFAULT_LOG_LEVEL


def write_log_level(base_dir: Path, name: str) -> str:
    """Validate *name* (one of :data:`~lifemodel.log.LOG_LEVEL_NAMES`,
    case-insensitive) and persist it, merged into the existing config so
    other keys are never clobbered.

    Raises :class:`ValueError` (via :func:`lifemodel.log.parse_log_level`) on
    an invalid name — nothing is read or written in that case. Returns the
    canonical lowercase name that was persisted.
    """
    parse_log_level(name)  # raises ValueError on an invalid name; validate first
    canonical = name.strip().lower()
    config = read_config(base_dir)
    config[_LOG_LEVEL_KEY] = canonical
    write_config(base_dir, config)
    return canonical


def set_log_level_for_dir(base_dir: Path, raw_args: str) -> str:
    """``/lifemodel loglevel [<level>]`` — the owner-facing command handler.

    No argument: report the current persisted level (never mutates — despite
    living under the ``loglevel`` subcommand, which is marked ``mutating`` in
    ``_SUBCOMMANDS`` because the WITH-argument form writes; matches the
    house pattern of a read/write pair sharing one subcommand, see ``set``
    vs. bare status elsewhere in this plugin). With an argument: validate
    against the 5 standard names, and on success persist it
    (:func:`write_log_level`) AND apply it at runtime
    (:func:`lifemodel.log.configure`) so the change takes effect immediately
    — not just on the next restart. An invalid name returns a readable usage
    message listing the valid names rather than raising (the command
    dispatch boundary in ``__init__.py`` already catches exceptions, but a
    clean usage message reads far better than the generic error wrapper for
    a plainly-mistyped level).
    """
    requested = raw_args.strip()
    current = read_log_level(base_dir)
    if not requested:
        return f"lifemodel loglevel: {current}\n"
    try:
        new_level = write_log_level(base_dir, requested)
    except ValueError:
        valid = ", ".join(LOG_LEVEL_NAMES)
        return (
            f"error: 'loglevel' invalid level {requested!r}. Valid levels: {valid}\n"
            f"usage: /lifemodel loglevel [{valid}]\n"
        )
    configure(parse_log_level(new_level))
    return f"lifemodel loglevel: {current} -> {new_level}\n"
