"""Make the flat-layout package importable as `lifemodel` in the dev checkout:
the repo directory IS the `lifemodel` package, so its parent must be on sys.path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
