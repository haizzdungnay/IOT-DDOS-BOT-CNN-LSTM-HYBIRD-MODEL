#!/usr/bin/env python3
"""
Script hợp nhất 4 file CSV từ Bot-IoT Dataset 5%
"""

import pandas as pd
import os
from pathlib import Path

# Đường dẫn các file
data_dir = r"E:\Bot_IOT_Dataset\Dataset\Dataset\5%\All features"
files = [
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv"
]

output_path = Path(data_dir) / "UNSW_2018_IoT_Botnet_Full5pc_Merged.csv"

print("=" * 60)
print("HỢP NHẤT DATASET BOT-IOT 5%")
print("=" * 60)

dfs = []
total_rows = 0

# Đọc từng file
for i, filename in enumerate(files, 1):
    filepath = Path(data_dir) / filename
    print(f"\n[{i}] Đọc {filename}...")
    
    df = pd.read_csv(filepath)
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    
    # Kiểm tra label column
    if 'label' in df.columns:
        normal = (df['label'] == 0).sum()
        attack = (df['label'] == 1).sum()
        print(f"    Normal: {normal:,}, Attack: {attack:,}")
    else:
        print(f"    Columns: {list(df.columns)}")
    
    dfs.append(df)
    total_rows += len(df)

# Hợp nhất
print(f"\n[Merge] Hợp nhất {len(dfs)} file...")
merged_df = pd.concat(dfs, ignore_index=True)

print(f"    Tổng rows: {total_rows:,}")
print(f"    Merged shape: {merged_df.shape}")

# Kiểm tra label
if 'label' in merged_df.columns:
    normal = (merged_df['label'] == 0).sum()
    attack = (merged_df['label'] == 1).sum()
    print(f"    Normal: {normal:,}, Attack: {attack:,}")
    print(f"    Tỉ lệ Attack: {attack/len(merged_df)*100:.2f}%")

# Lưu
print(f"\n[Save] Lưu vào: {output_path}")
merged_df.to_csv(output_path, index=False)
print(f"✅ Hoàn thành! File: {output_path}")
print(f"    Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
