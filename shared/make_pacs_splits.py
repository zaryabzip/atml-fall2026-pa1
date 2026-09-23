"""Create the stratified 80/20 source splits (seed 6304). Sketch is not included.

    python -m shared.make_pacs_splits --pacs-root /path/to/PACS
"""
import argparse

from common.io import data_root
from shared.pacs_protocol import SPLIT_FILE, make_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs-root", default=str(data_root() / "pacs"))
    args = parser.parse_args()
    split = make_splits(args.pacs_root)
    for domain, parts in split["sources"].items():
        print(f"{domain}: train={len(parts['train'])} val={len(parts['val'])}")
    print(f"Saved {SPLIT_FILE}")


if __name__ == "__main__":
    main()
