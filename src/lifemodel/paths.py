"""Per-profile filesystem paths for the lifemodel plugin.

Pure path arithmetic only — this module imports **nothing** from Hermes, so it
stays importable and unit-testable in isolation. The adapter layer
(:func:`lifemodel.register`) resolves the active profile home via the host's
``get_hermes_home()`` and passes it in here.

One creature ≈ one profile (HLA §4, D3/D5): state lives under the profile home
so a profile backup captures the whole being. This module only *computes* the
path; nothing is created on disk here — state files land in task 0.2.
"""

from __future__ import annotations

from pathlib import Path

#: Directory name for all lifemodel state, created under the profile home.
STATE_DIR_NAME = "lifemodel"


def state_dir(profile_home: Path) -> Path:
    """Return the lifemodel state directory under *profile_home*.

    *profile_home* is the active Hermes profile home (``get_hermes_home()``),
    so the result is per-profile. This resolves the path only; creation is
    deferred to task 0.2.
    """
    return profile_home / STATE_DIR_NAME
