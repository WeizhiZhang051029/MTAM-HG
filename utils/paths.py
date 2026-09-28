from __future__ import annotations

from pathlib import Path

import config


def project_path(path: str | Path, *, resolve: bool = False) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else config.PROJECT_ROOT / raw
    return resolved.resolve() if resolve else resolved
