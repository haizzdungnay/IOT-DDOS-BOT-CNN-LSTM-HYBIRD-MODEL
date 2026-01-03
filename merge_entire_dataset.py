#!/usr/bin/env python3
"""
Script hợp nhất dataset Entire Bot-IoT và lấy tất cả features có sẵn
Tự động khám phá cấu trúc dữ liệu
"""

import pandas as pd
import os
from pathlib import Path
import json

# Đường dẫn
entire_dir = r"E:\Bot_IOT_Dataset\Dataset\Dataset\Entire Dataset"

# Lấy danh sách file
csv_files = sorted([f for f in os.listdir(entire_dir) if f.endswith('.csv') and 'Dataset_' in f])

print("=" * 60)
print("KHÁM PHÁ VÀ HỢP NHẤT DATASET ENTIRE BOT-IOT")
print("=" * 60)
print(f"\n[INFO] Tìm được {len(csv_files)} file CSV")

if not csv_files:
    print("❌ Không tìm thấy file CSV!")
    exit(1)

# Đọc file đầu tiên để khám phá cấu trúc
first_file = csv_files[0]
print(f"\n[1] Khám phá cấu trúc từ {first_file}...")

df_sample = pd.read_csv(Path(entire_dir) / first_file, nrows=1000)
print(f"    Columns ({len(df_sample.columns)}):")
for i, col in enumerate(df_sample.columns, 1):
    print(f"      {i:2}. {col}")

# Tìm label column
label_col = None
for name in ['attack', 'label', 'Label', 'ATTACK', 'Attack', 'class']:
    if name in df_sample.columns:
        label_col = name
        break

if label_col:
    print(f"\n✅ Tìm thấy label column: '{label_col}'")
else:
    print(f"\n⚠️  Không tìm thấy label column! Dùng 'attack'")
    label_col = 'attack'

# Lấy tất cả feature (trừ label)
all_features = [col for col in df_sample.columns if col != label_col]
print(f"\n📊 Tổng features: {len(all_features)}")

# Hợp nhất (lấy subset để nhanh)
print(f"\n[2] Hợp nhất {len(csv_files)} files...")
print("    ⏳ Đang đọc... (mỗi file ~200-250MB)")

dfs = []
total_normal = 0
total_attack = 0

for i, filename in enumerate(csv_files, 1):
    if i % 10 == 0 or i == len(csv_files):
        print(f"    [{i}/{len(csv_files)}] {filename}")
    
    filepath = Path(entire_dir) / filename
    try:
        df = pd.read_csv(filepath)
        
        # Thống kê
        if label_col in df.columns:
            n = (df[label_col] == 0).sum()
            a = (df[label_col] == 1).sum()
            total_normal += n
            total_attack += a
        
        dfs.append(df)
    except Exception as e:
        print(f"    ⚠️  Lỗi đọc {filename}: {e}")

print(f"\n    Đã load {len(dfs)} files")
print(f"    Normal: {total_normal:,}")
print(f"    Attack: {total_attack:,}")

# Hợp nhất
print(f"\n[3] Hợp nhất {len(dfs)} files thành 1 DataFrame...")
merged_df = pd.concat(dfs, ignore_index=True)

print(f"    Shape: {merged_df.shape}")
print(f"    Features: {len(merged_df.columns)}")

# Lưu thông tin cấu hình
config_data = {
    "features": all_features,
    "label_column": label_col,
    "total_samples": len(merged_df),
    "normal_count": total_normal,
    "attack_count": total_attack,
    "columns": list(merged_df.columns)
}

config_path = Path(entire_dir) / "dataset_config.json"
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)
print(f"\n✅ Lưu config: {config_path}")

# Lưu merged
output_path = Path(entire_dir) / "UNSW_2018_IoT_Botnet_Entire_Merged.csv"
print(f"\n[4] Lưu merged file...")
print(f"    {output_path}")
print("    ⏳ Đang ghi file... (mất ~5-10 phút)")

merged_df.to_csv(output_path, index=False)

file_size_gb = os.path.getsize(output_path) / 1024 / 1024 / 1024
print(f"\n✅ Hoàn thành!")
print(f"    File size: {file_size_gb:.2f} GB")
print(f"    Total rows: {len(merged_df):,}")
print(f"    Normal: {total_normal:,} ({total_normal/len(merged_df)*100:.2f}%)")
print(f"    Attack: {total_attack:,} ({total_attack/len(merged_df)*100:.2f}%)")

print(f"\n📌 Bước tiếp theo:")
print(f"   1. Cập nhật training/config.py với features mới")
print(f"   2. Training: python training/train_all.py --data \"{output_path}\" --models LSTM HYBRID")
