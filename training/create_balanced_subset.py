#!/usr/bin/env python3
"""
Create a balanced train/val/test subset from a large Bot-IoT CSV using streaming.
- Avoids loading the full file in memory by sampling per chunk.
- Targets an equal number of Normal and Attack samples for more meaningful metrics.
"""

import argparse
import os
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create balanced subset from Bot-IoT CSV")
    parser.add_argument("--source", required=True, help="Path to merged CSV containing 'attack' column")
    parser.add_argument("--output_dir", default="training/outputs/balanced", help="Directory to save splits")
    parser.add_argument("--label_col", default="attack", help="Label column name")
    parser.add_argument("--normal_target", type=int, default=50000, help="Target number of Normal samples")
    parser.add_argument("--attack_target", type=int, default=50000, help="Target number of Attack samples")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunksize for streaming")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio from selected data")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio from selected data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def limited_append(target_list: List[pd.DataFrame], df: pd.DataFrame, limit: int):
    """Append rows to target_list without exceeding limit."""
    remaining = limit - sum(len(x) for x in target_list)
    if remaining <= 0:
        return
    if len(df) > remaining:
        df = df.sample(n=remaining, random_state=args.seed)
    target_list.append(df)


def main(args):
    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normals: List[pd.DataFrame] = []
    attacks: List[pd.DataFrame] = []

    total_norm = total_att = 0

    reader = pd.read_csv(src, chunksize=args.chunksize, low_memory=False)
    for idx, chunk in enumerate(reader, 1):
        if args.label_col not in chunk.columns:
            raise KeyError(f"Label column '{args.label_col}' not found in chunk {idx}")

        chunk_norm = chunk[chunk[args.label_col] == 0]
        chunk_att = chunk[chunk[args.label_col] == 1]

        total_norm += len(chunk_norm)
        total_att += len(chunk_att)

        limited_append(normals, chunk_norm, args.normal_target)
        limited_append(attacks, chunk_att, args.attack_target)

        have_norm = sum(len(x) for x in normals)
        have_att = sum(len(x) for x in attacks)

        print(f"Chunk {idx:03d}: keep N={have_norm}/{args.normal_target}, A={have_att}/{args.attack_target}")

        if have_norm >= args.normal_target and have_att >= args.attack_target:
            break

    if sum(len(x) for x in normals) == 0 or sum(len(x) for x in attacks) == 0:
        raise RuntimeError("Insufficient samples collected for one or both classes. Increase targets or ensure data availability.")

    df_norm = pd.concat(normals, ignore_index=True)
    df_att = pd.concat(attacks, ignore_index=True)

    df_balanced = pd.concat([df_norm, df_att], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"Collected balanced set: total={len(df_balanced)}, Normal={len(df_norm)}, Attack={len(df_att)}")

    test_ratio = args.test_ratio
    val_ratio = args.val_ratio
    train_ratio = 1.0 - test_ratio - val_ratio
    if train_ratio <= 0:
        raise ValueError("train_ratio must be positive; adjust val_ratio and test_ratio")

    train_df, temp_df = train_test_split(df_balanced, test_size=(1 - train_ratio), stratify=df_balanced[args.label_col], random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + val_ratio), stratify=temp_df[args.label_col], random_state=args.seed)

    train_path = out_dir / "train_balanced.csv"
    val_path = out_dir / "val_balanced.csv"
    test_path = out_dir / "test_balanced.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Val:   {val_path} ({len(val_df)} rows)")
    print(f"  Test:  {test_path} ({len(test_df)} rows)")

    # Quick label counts
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = (df[args.label_col] == 0).sum()
        a = (df[args.label_col] == 1).sum()
        print(f"  {name}: N={n} A={a}")

    # Minimal class weights suggestion
    total = len(df_balanced)
    class_weights = {
        0: total / (2 * len(df_norm)),
        1: total / (2 * len(df_att)),
    }
    weights_path = out_dir / "class_weights.json"
    pd.Series(class_weights).to_json(weights_path, indent=2)
    print(f"Class weights saved to {weights_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
