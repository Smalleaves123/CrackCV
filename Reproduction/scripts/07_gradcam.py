from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT
from Reproduction.src.explain.grad_cam import generate_gradcam
from Reproduction.src.utils.config import attach_project_root, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = attach_project_root(load_config(args.config), ROOT)
    config.setdefault("weights", {})
    config["weights"]["pretrained"] = False
    config["weights"]["offline_mode"] = True
    out = args.out or str(Path(config["runtime"]["run_dir"]) / "gradcam" / (Path(args.image).stem + "_gradcam.png"))
    print(json.dumps({"output": generate_gradcam(config, args.checkpoint, args.image, out)}, indent=2))


if __name__ == "__main__":
    main()
