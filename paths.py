"""Per-profile filesystem paths for the lifemodel plugin.

Pure path arithmetic only — this module imports **nothing** from Hermes, so it
stays importable and unit-testable in isolation. The adapter layer
(:func:`lifemodel.register`) resolves the active profile home via the host's
``get_hermes_home()`` and passes it in here.

One creature ≈ one profile (HLA §4, D3/D5): state lives under the profile home
so a profile backup captures the whole being. Following the established Hermes
plugin convention (see the nelix plugin), per-plugin data is kept workspace-
scoped — i.e. under ``<profile_home>/workspace/lifemodel/`` — rather than
dropped directly in the profile-home root next to ``config.yaml``, ``.env``,
``cron/``, and ``plugins/``. This module only *computes* the path; nothing is
created on disk here — state files land in task 0.2.
"""

from __future__ import annotations

from pathlib import Path

#: Directory name grouping all per-plugin data under the profile home. Each
#: plugin owns a subdir of this (e.g. ``workspace/lifemodel/``), matching the
#: nelix convention, so the profile-home root stays clean.
WORKSPACE_DIR_NAME = "workspace"

#: Directory name for all lifemodel state, created under the profile home's
#: ``workspace/`` subtree.
STATE_DIR_NAME = "lifemodel"


def state_dir(profile_home: Path) -> Path:
    """Return the lifemodel state directory under *profile_home*.

    *profile_home* is the active Hermes profile home (``get_hermes_home()``),
    so the result is per-profile. The state dir is workspace-scoped —
    ``<profile_home>/workspace/lifemodel/`` — keeping plugin data out of the
    profile-home root. This resolves the path only; creation is deferred to
    task 0.2.
    """
    return profile_home / WORKSPACE_DIR_NAME / STATE_DIR_NAME
