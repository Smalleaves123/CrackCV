from __future__ import annotations

import shutil
from pathlib import Path


def count_files(root: Path) -> int:
    return sum(1 for item in root.rglob("*") if item.is_file())


def backup_dataset(src: str | Path, dst: str | Path, force: bool = False) -> dict:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_path}")
    if dst_path.exists() and not force:
        return {
            "status": "skipped",
            "src_files": count_files(src_path),
            "dst_files": count_files(dst_path),
            "dst": str(dst_path),
        }
    if dst_path.exists() and force:
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)
    return {
        "status": "copied",
        "src_files": count_files(src_path),
        "dst_files": count_files(dst_path),
        "dst": str(dst_path),
    }
