from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


PathLike = Union[str, Path]
CLASS_NAMES = ["non-crack", "crack"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
RAW_TO_CANONICAL = {
    "Negative": "non-crack",
    "Positive": "crack",
    "negative": "non-crack",
    "positive": "crack",
    "non-crack": "non-crack",
    "crack": "crack",
}


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: Dict[str, Any], path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: PathLike) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
