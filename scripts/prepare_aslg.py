"""Download ASLG-PC12 (English <-> ASL gloss, ~87k sentence pairs) into data/aslg.

    python scripts/prepare_aslg.py

The copy used here is the train/dev/test split published with Yin & Read (2020),
"Better Sign Language Translation with STMC-Transformer" (github.com/kayoyin/transformer-slt).

Caveat: ASLG-PC12 gloss was generated from English (Europarl) by rules, not written by
signers. It is fine for proving the model works, but it is NOT ground truth ASL.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/kayoyin/transformer-slt"
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=ROOT / "data" / "aslg", type=Path)
    p.add_argument("--source", type=Path, help="existing transformer-slt checkout (skips the git clone)")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        source = args.source
        if source is None:
            source = Path(tmp) / "transformer-slt"
            subprocess.run(["git", "clone", "--depth", "1", REPO, str(source)], check=True)
        args.out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            shutil.copy(source / "data" / f"aslg.{split}.en", args.out / f"{split}.en")
            shutil.copy(source / "data" / f"aslg.{split}.gloss.asl", args.out / f"{split}.gloss")
    for f in sorted(args.out.iterdir()):
        print(f"{f}: {sum(1 for _ in open(f, encoding='utf-8-sig'))} lines")


if __name__ == "__main__":
    main()
