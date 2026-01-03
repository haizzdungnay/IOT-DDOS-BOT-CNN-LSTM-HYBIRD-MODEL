#!/usr/bin/env python3
"""
Script hợp nhất dataset Entire Bot-IoT với xử lý theo chunk
Tiết kiệm RAM bằng cách ghi trực tiếp ra file
"""

import pandas as pd
import os
from pathlib import Path
import json

# Đường dẫn
entire_dir = r"E:\Bot_IOT_Dataset\Dataset\Dataset\Entire Dataset"
output_path = Path(entire_dir) / "UNSW_2018_IoT_Botnet_Entire_Merged.csv"

# Lấy danh sách file
csv_files = sorted([f for f in os.listdir(entire_dir) if f.endswith('.csv') and 'Dataset_' in f])

print("=" * 60)
print("HỢP NHẤT DATASET ENTIRE BOT-IOT (TIẾT KIỆM RAM)")
print("=" * 60)
print(f"\n[INFO] Tìm được {len(csv_files)} file CSV")

if not csv_files:
    print("❌ Không tìm thấy file CSV!")
    exit(1)

# Khám phá cấu trúc từ file đầu tiên
first_file = csv_files[0]
print(f"\n[1] Khám phá cấu trúc từ {first_file}...")

df_sample = pd.read_csv(Path(entire_dir) / first_file, nrows=100)
print(f"    Columns: {len(df_sample.columns)}")
print(f"    Columns names: {list(df_sample.columns)[:10]}...")

# Xác định label column
label_col = 'attack'
all_features = list(df_sample.columns)

print(f"    Label column: '{label_col}'")
print(f"    Total columns: {len(all_features)}")

# Hợp nhất từng file và ghi ra (streaming)
print(f"\n[2] Hợp nhất {len(csv_files)} files (ghi trực tiếp)...")
print("    Method: Chunk-by-chunk (tiết kiệm RAM)")

total_normal = 0
total_attack = 0
total_rows = 0
first_write = True

for i, filename in enumerate(csv_files, 1):
    filepath = Path(entire_dir) / filename

    try:
        print(f"    [{i:2}/{len(csv_files)}] Xử lý {filename}...", flush=True)

        # Đọc theo chunk để giảm RAM
        chunk_iter = pd.read_csv(filepath, chunksize=200_000, low_memory=False)
        file_normal = 0
        file_attack = 0
        file_rows = 0

        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            n = (chunk[label_col] == 0).sum() if label_col in chunk.columns else 0
            a = (chunk[label_col] == 1).sum() if label_col in chunk.columns else 0

            total_normal += n
            total_attack += a
            total_rows += len(chunk)

            file_normal += n
            file_attack += a
            file_rows += len(chunk)

            mode = 'w' if first_write else 'a'
            header = first_write
            chunk.to_csv(output_path, mode=mode, header=header, index=False)
            first_write = False

            print(f"        - Chunk {chunk_idx:02d}: rows={len(chunk):,} (N:{n:,} A:{a:,})", flush=True)

        print(f"      ✓ Done {filename} (rows:{file_rows:,} N:{file_normal:,} A:{file_attack:,})")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

print(f"\n[3] Kết quả hợp nhất:")
print(f"    Total rows: {total_rows:,}")
print(f"    Normal: {total_normal:,} ({total_normal/total_rows*100:.3f}%)")
print(f"    Attack: {total_attack:,} ({total_attack/total_rows*100:.3f}%)")

file_size_gb = os.path.getsize(output_path) / 1024 / 1024 / 1024
print(f"\n    Output file: {output_path}")
print(f"    File size: {file_size_gb:.2f} GB")

# Lưu config
config_data = {
    "features": all_features,
    "label_column": label_col,
    "total_samples": total_rows,
    "normal_count": total_normal,
    "attack_count": total_attack,
    "normal_ratio": total_normal/total_rows,
    "attack_ratio": total_attack/total_rows
}

config_path = Path(entire_dir) / "dataset_config.json"
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\n✅ Hoàn thành!")
print(f"   Config: {config_path}")
print(f"\n📌 Lệnh training tiếp theo:")
print(f'   cd training')
print(f'   python train_all.py --data "{output_path}" --models LSTM --epochs 30')
