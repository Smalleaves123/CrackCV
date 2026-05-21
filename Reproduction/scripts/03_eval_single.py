from __future__ import annotations

import argparse
import json

from common import ROOT
from Reproduction.src.engine.evaluator import evaluate
from Reproduction.src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("weights", {})
    config["weights"]["pretrained"] = False
    config["weights"]["offline_mode"] = True
    if "run_dir" not in config.get("runtime", {}):
        config["runtime"]["run_dir"] = str(ROOT / "Reproduction" / "results" / config["model"]["name"] / config["runtime"]["strategy"])
    print(json.dumps(evaluate(config, args.checkpoint, split=args.split), indent=2))


if __name__ == "__main__":
    main()
