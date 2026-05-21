from __future__ import annotations

import argparse
import json

from common import ROOT
from Reproduction.src.data.backup_dataset import backup_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "dataset"))
    parser.add_argument("--dst", default=str(ROOT / "data_work" / "raw_backup"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(backup_dataset(args.src, args.dst, force=args.force), indent=2))


if __name__ == "__main__":
    main()
