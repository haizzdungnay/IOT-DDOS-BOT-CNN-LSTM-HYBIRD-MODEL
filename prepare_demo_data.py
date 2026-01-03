"""
Prepare Demo Test Data
=======================
Tạo file demo_test.csv từ test set với tỷ lệ balanced
để demo hiệu quả trên web.

Sẽ lấy:
- 500 Normal samples
- 500 Attack samples  
→ Total: 1000 samples cho demo

Author: IoT Security Research Team
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
import os
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
RAW_CSV = r"D:\Project\IoT\Dataset\EntireDataset\botiot.csv"
OUTPUT_CSV = "data/demo_test.csv"

# Feature names (Bot-IoT) - 15 features cần cho model
FEATURE_NAMES = [
    'pkts', 'bytes', 'dur', 'mean', 'stddev', 'sum', 'min', 'max',
    'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate'
]

print("="*70)
print("PREPARING DEMO TEST DATA")
print("="*70)

# Create data directory
os.makedirs('data', exist_ok=True)

print("\n[1/4] Loading Bot-IoT Dataset...")
print(f"   Reading from: {RAW_CSV}")
print(f"   Note: Collecting from entire dataset (Normal samples are rare!)")

# Load dataset in chunks to avoid memory issue
chunk_size = 100000
normal_samples = []
attack_samples = []
normal_target = 500
attack_target = 500
current_row = 0
chunks_processed = 0

for chunk in pd.read_csv(RAW_CSV, low_memory=False, chunksize=chunk_size):
    current_row += len(chunk)
    chunks_processed += 1
    
    # Collect Normal (priority - rare!)
    if sum([len(df) for df in normal_samples]) < normal_target:
        normal_chunk = chunk[chunk['attack'] == 0]
        if len(normal_chunk) > 0:
            normal_samples.append(normal_chunk)
            total_normal = sum([len(df) for df in normal_samples])
            print(f"   Normal: {total_normal}/{normal_target} (found in chunk {chunks_processed}, row ~{current_row:,})")
    
    # Collect Attack  
    if sum([len(df) for df in attack_samples]) < attack_target:
        attack_chunk = chunk[chunk['attack'] == 1]
        if len(attack_chunk) > 0:
            attack_samples.append(attack_chunk)
            total_attack = sum([len(df) for df in attack_samples])
            if chunks_processed <= 5 or total_attack <= attack_target:  # Only print first few chunks or when reaching target
                print(f"   Attack: {total_attack}/{attack_target}")
    
    # Stop when we have enough
    if (sum([len(df) for df in normal_samples]) >= normal_target and 
        sum([len(df) for df in attack_samples]) >= attack_target):
        print(f"\n   ✅ Collected enough samples from {chunks_processed} chunks (~{current_row:,} rows)")
        break

print("\n[2/4] Combining samples...")

# Check if we collected enough
if len(normal_samples) == 0:
    print("   ❌ ERROR: No Normal samples found!")
    print("   The dataset might not have 'attack' column or all values are 1")
    sys.exit(1)

if len(attack_samples) == 0:
    print("   ❌ ERROR: No Attack samples found!")
    sys.exit(1)

# Combine
df_normal = pd.concat(normal_samples, ignore_index=True).head(normal_target)
df_attack = pd.concat(attack_samples, ignore_index=True).head(attack_target)

print(f"   Normal: {len(df_normal)}")
print(f"   Attack: {len(df_attack)}")

# Merge
df_demo = pd.concat([df_normal, df_attack], ignore_index=True)

print(f"\n   Total samples: {len(df_demo)}")

print("\n[3/4] Sorting by time...")
# Sort by time if column exists
if 'stime' in df_demo.columns:
    df_demo = df_demo.sort_values('stime').reset_index(drop=True)
else:
    print("   Warning: 'stime' column not found, skipping sort")

print("\n[4/4] Saving to CSV...")

# Check if columns exist, rename if present
rename_dict = {}
if 'saddr' in df_demo.columns:
    rename_dict['saddr'] = 'srcip'
if 'daddr' in df_demo.columns:
    rename_dict['daddr'] = 'dstip'

if rename_dict:
    df_demo = df_demo.rename(columns=rename_dict)

# Select columns for demo - only use columns that exist
output_columns = FEATURE_NAMES + ['attack']

# Add metadata if available
if 'stime' in df_demo.columns:
    output_columns.append('stime')
if 'srcip' in df_demo.columns:
    output_columns.append('srcip')
if 'dstip' in df_demo.columns:
    output_columns.append('dstip')

# Filter only existing columns
existing_columns = [col for col in output_columns if col in df_demo.columns]

# Save only existing columns
output_df = df_demo[existing_columns]
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"   ✅ Saved to: {OUTPUT_CSV}")
print(f"   Rows: {len(output_df)}")
print(f"   Columns: {len(output_df.columns)}")
print(f"   Column names: {list(output_df.columns)}")

print("\n" + "="*70)
print("✅ DEMO DATA READY!")
print("="*70)
print("\nStatistics:")
print(f"  Normal:  {(output_df['attack'] == 0).sum():,} samples ({(output_df['attack'] == 0).sum()/len(output_df)*100:.1f}%)")
print(f"  Attack:  {(output_df['attack'] == 1).sum():,} samples ({(output_df['attack'] == 1).sum()/len(output_df)*100:.1f}%)")
print(f"\nFile size: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.2f} MB")
print("\nYou can now run: python app.py")
print("="*70)
