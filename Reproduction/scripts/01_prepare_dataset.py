from __future__ import annotations

import argparse
import json

from common import ROOT
from Reproduction.src.data.prepare_dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "data_work" / "raw_backup"))
    parser.add_argument("--out", default=str(ROOT / "data_work" / "processed"))
    parser.add_argument("--splits-out", default=str(ROOT / "data_work" / "splits"))
    parser.add_argument("--image-size", type=int, default=227)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.src, args.out, args.splits_out, args.image_size, args.seed), indent=2))


if __name__ == "__main__":
    main()
