"""Downloads CNN/DailyMail (via Hugging Face `datasets`) and exports
train/val/test CSVs into `data/`, matching this project's expected
`article` / `highlights` column convention.

The full dataset is ~287k/13k/11k (train/val/test) examples — large
enough that a single CPU/laptop training run would take a very long
time. By default this script caps each split so a full run finishes in
a reasonable amount of time; pass 0 to disable a cap and use the full
split.

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --max_train 5000 --max_val 500 --max_test 500
    python scripts/download_dataset.py --max_train 0   # full train split, no cap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CNN/DailyMail and export train/val/test CSVs")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--version", type=str, default="3.0.0")
    parser.add_argument("--max_train", type=int, default=20000, help="Cap on train examples (0 = no cap, full ~287k)")
    parser.add_argument("--max_val", type=int, default=2000, help="Cap on validation examples (0 = no cap, full ~13k)")
    parser.add_argument("--max_test", type=int, default=2000, help="Cap on test examples (0 = no cap, full ~11k)")
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "data"))
    return parser.parse_args()


def export_split(ds_split, cap: int, out_path: Path) -> None:
    if cap:
        ds_split = ds_split.select(range(min(cap, len(ds_split))))

    df = pd.DataFrame({"article": ds_split["article"], "highlights": ds_split["highlights"]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} examples to {out_path}")


def main() -> None:
    args = parse_args()

    # Imported here so `--help` works without the (heavier, slower-importing)
    # datasets library installed.
    from datasets import load_dataset

    print(f"Downloading {args.dataset} ({args.version})... this can take a few minutes on first run.")
    ds = load_dataset(args.dataset, args.version)

    out_dir = Path(args.out_dir)
    export_split(ds["train"], args.max_train, out_dir / "train.csv")
    export_split(ds["validation"], args.max_val, out_dir / "val.csv")
    export_split(ds["test"], args.max_test, out_dir / "test.csv")

    print("Done. Next: python -m src.training.train --train_path data/train.csv --val_path data/val.csv")


if __name__ == "__main__":
    main()
