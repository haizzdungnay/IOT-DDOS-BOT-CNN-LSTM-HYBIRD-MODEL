#!/usr/bin/env python3
"""
Script hợp nhất 4 file CSV từ Bot-IoT Dataset 5%
Chỉ lấy 15 features cần dùng để giảm dung lượng
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

# 15 features + label
FEATURES = ['pkts', 'bytes', 'dur', 'mean', 'stddev', 'sum', 'min', 'max', 
            'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate']
LABEL = 'attack'

output_path = Path(data_dir) / "UNSW_2018_IoT_Botnet_Full5pc_Merged_Optimized.csv"

print("=" * 60)
print("HỢP NHẤT & TỐI ƯU DATASET BOT-IOT 5%")
print("=" * 60)

dfs = []
total_rows = 0

# Đọc từng file (chỉ lấy features cần thiết)
for i, filename in enumerate(files, 1):
    filepath = Path(data_dir) / filename
    print(f"\n[{i}] Đọc {filename}...")
    
    df = pd.read_csv(filepath, usecols=FEATURES + [LABEL])
    print(f"    Rows: {len(df):,}")
    
    # Kiểm tra label
    normal = (df[LABEL] == 0).sum()
    attack = (df[LABEL] == 1).sum()
    print(f"    Normal: {normal:,}, Attack: {attack:,}")
    
    dfs.append(df)
    total_rows += len(df)

# Hợp nhất
print(f"\n[Merge] Hợp nhất {len(dfs)} file...")
merged_df = pd.concat(dfs, ignore_index=True)

print(f"    Tổng rows: {total_rows:,}")
print(f"    Merged shape: {merged_df.shape}")

# Kiểm tra label
normal = (merged_df[LABEL] == 0).sum()
attack = (merged_df[LABEL] == 1).sum()
print(f"    Normal: {normal:,}, Attack: {attack:,}")
print(f"    Tỉ lệ Attack: {attack/len(merged_df)*100:.2f}%")

# Lưu
print(f"\n[Save] Lưu vào: {output_path}")
print("    Đang ghi file... (có thể mất vài phút)")
merged_df.to_csv(output_path, index=False)

file_size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"✅ Hoàn thành!")
print(f"    File: {output_path}")
print(f"    Size: {file_size_mb:.1f} MB")
print(f"\n📌 Sẵn sàng train với lệnh:")
print(f'    python training/train_all.py --data "{output_path}" --models LSTM --epochs 30')
