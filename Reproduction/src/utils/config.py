from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if base_path is not None:
        with Path(base_path).open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle) or {}
        config = deep_update(base, config)
    return config


def attach_project_root(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    config.setdefault("runtime", {})
    config["runtime"]["project_root"] = str(Path(project_root).resolve())
    return config


def resolve_config_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = config.get("runtime", {}).get("project_root")
    if project_root:
        return Path(project_root) / path
    return path


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = deepcopy(config)
    runtime = serializable.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("project_root", None)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
