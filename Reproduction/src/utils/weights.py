from __future__ import annotations

from pathlib import Path


def resolve_local_weights(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    return str(candidate) if candidate.exists() else None
