from __future__ import annotations

import argparse
import json

from common import ROOT
from Reproduction.src.engine.predictor import predict_image
from Reproduction.src.utils.config import attach_project_root, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    config = attach_project_root(load_config(args.config), ROOT)
    config.setdefault("weights", {})
    config["weights"]["pretrained"] = False
    config["weights"]["offline_mode"] = True
    print(json.dumps(predict_image(config, args.checkpoint, args.image), indent=2))


if __name__ == "__main__":
    main()
