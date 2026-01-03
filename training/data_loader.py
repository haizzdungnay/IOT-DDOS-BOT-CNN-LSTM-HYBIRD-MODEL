"""
=============================================================================
DATA LOADER - Load va xu ly du lieu thong nhat
=============================================================================
Muc dich:
- Load Bot-IoT dataset
- Chuan hoa du lieu (StandardScaler)
- Tao sequences voi sliding window
- Chia train/val/test DONG NHAT cho tat ca models
- Luu lai X_test, y_test de share giua cac thanh vien

Author: IoT Security Research Team
Date: 2026-01-03
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json

from config import (
    KEEP_FEATURES, LABEL_COLUMN, N_FEATURES,
    TIME_STEPS, STRIDE, BATCH_SIZE,
    RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    OUTPUTS_DIR, PROCESSED_DATA_DIR, MODELS_DIR, DEVICE
)


class BotIoTDataset(Dataset):
    """
    PyTorch Dataset cho Bot-IoT
    Toi uu cho GPU voi pin_memory
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features array (n_samples, time_steps, n_features)
            y: Labels array (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(X: np.ndarray, y: np.ndarray,
                     time_steps: int = TIME_STEPS,
                     stride: int = STRIDE) -> tuple:
    """
    Tao sequences tu du lieu voi sliding window

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        time_steps: So buoc thoi gian
        stride: Buoc nhay

    Returns:
        X_seq: (n_sequences, time_steps, n_features)
        y_seq: (n_sequences,)
    """
    X_seq, y_seq = [], []

    for i in range(0, len(X) - time_steps + 1, stride):
        X_seq.append(X[i:i + time_steps])
        # Lay label cua timestep cuoi cung
        y_seq.append(y[i + time_steps - 1])

    return np.array(X_seq), np.array(y_seq)


def load_and_preprocess_data(csv_path: str,
                              save_scaler: bool = True,
                              save_test_set: bool = True) -> dict:
    """
    Load va xu ly du lieu Bot-IoT

    QUAN TRONG: Ham nay dam bao tat ca thanh vien dung chung:
    - Cung RANDOM_STATE de chia data
    - Cung scaler de chuan hoa
    - Cung test set de danh gia

    Args:
        csv_path: Duong dan den file CSV
        save_scaler: Luu scaler de dung cho inference
        save_test_set: Luu test set de chia se

    Returns:
        dict: Chua DataLoaders va thong tin
    """
    print("=" * 60)
    print("LOAD VA XU LY DU LIEU")
    print("=" * 60)

    # =========================================================================
    # 1. Load CSV
    # =========================================================================
    print(f"\n[1] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"    Tong so mau: {len(df):,}")

    # =========================================================================
    # 2. Trich xuat features va labels
    # =========================================================================
    print(f"\n[2] Trich xuat {N_FEATURES} features...")
    X = df[KEEP_FEATURES].values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.float32)

    print(f"    Normal (0): {(y == 0).sum():,}")
    print(f"    Attack (1): {(y == 1).sum():,}")
    print(f"    Ti le Attack: {(y == 1).mean() * 100:.2f}%")

    # =========================================================================
    # 3. Chia train/test TRUOC KHI tao sequences
    # =========================================================================
    print(f"\n[3] Chia train/test (random_state={RANDOM_STATE})...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Dam bao ty le Normal/Attack dong deu
    )
    print(f"    Train: {len(X_train_raw):,} mau")
    print(f"    Test:  {len(X_test_raw):,} mau")

    # =========================================================================
    # 4. Chuan hoa du lieu (fit tren train, transform ca train va test)
    # =========================================================================
    print("\n[4] Chuan hoa du lieu (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if save_scaler:
        # Save to MODELS_DIR for replay_detector compatibility
        scaler_path = MODELS_DIR / "scaler_standard.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"    Saved scaler: {scaler_path}")

    # =========================================================================
    # 5. Tao sequences
    # =========================================================================
    print(f"\n[5] Tao sequences (time_steps={TIME_STEPS}, stride={STRIDE})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw)

    print(f"    Train sequences: {len(X_train_seq):,}")
    print(f"    Test sequences:  {len(X_test_seq):,}")

    # =========================================================================
    # 6. Chia validation tu train
    # =========================================================================
    print(f"\n[6] Chia validation ({VAL_SIZE * 100}% tu train)...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_seq, y_train_seq,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_seq
    )
    print(f"    Train final: {len(X_train_final):,}")
    print(f"    Validation:  {len(X_val):,}")
    print(f"    Test:        {len(X_test_seq):,}")

    # =========================================================================
    # 7. Luu test set de chia se giua cac thanh vien
    # =========================================================================
    if save_test_set:
        print("\n[7] Luu test set de dong bo...")
        np.save(OUTPUTS_DIR / "X_test.npy", X_test_seq)
        np.save(OUTPUTS_DIR / "y_test.npy", y_test_seq)
        print(f"    Saved: X_test.npy ({X_test_seq.shape})")
        print(f"    Saved: y_test.npy ({y_test_seq.shape})")

        # Luu metadata
        metadata = {
            "n_samples_train": len(X_train_final),
            "n_samples_val": len(X_val),
            "n_samples_test": len(X_test_seq),
            "time_steps": TIME_STEPS,
            "n_features": N_FEATURES,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "features": KEEP_FEATURES,
            "attack_ratio_test": float((y_test_seq == 1).mean())
        }
        with open(OUTPUTS_DIR / "data_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"    Saved: data_metadata.json")

    # =========================================================================
    # 8. Tao DataLoaders (toi uu cho GPU)
    # =========================================================================
    print("\n[8] Tao DataLoaders...")

    # Kiem tra co GPU khong de bat pin_memory
    use_cuda = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0  # Parallel data loading
    pin_memory = use_cuda  # Faster GPU transfer

    train_dataset = BotIoTDataset(X_train_final, y_train_final)
    val_dataset = BotIoTDataset(X_val, y_val)
    test_dataset = BotIoTDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Bo batch cuoi neu khong du
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")

    print("\n" + "=" * 60)
    print("HOAN THANH XU LY DU LIEU!")
    print("=" * 60)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "metadata": metadata if save_test_set else None
    }


def load_test_set_only(test_dir: str = None) -> tuple:
    """
    Load chi test set (dung khi chi can danh gia)

    QUAN TRONG: Dung ham nay de dam bao tat ca thanh vien
    dung CUNG MOT test set

    Args:
        test_dir: Thu muc chua X_test.npy va y_test.npy

    Returns:
        X_test, y_test, test_loader
    """
    if test_dir is None:
        test_dir = OUTPUTS_DIR

    test_dir = Path(test_dir)

    print("Loading test set...")
    X_test = np.load(test_dir / "X_test.npy")
    y_test = np.load(test_dir / "y_test.npy")

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Attack ratio: {(y_test == 1).mean() * 100:.2f}%")

    # Tao DataLoader
    use_cuda = torch.cuda.is_available()
    test_dataset = BotIoTDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda
    )

    return X_test, y_test, test_loader


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_csv>")
        print("Example: python data_loader.py ../data/botiot.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    data = load_and_preprocess_data(csv_path)

    print("\n[TEST] Kiem tra 1 batch:")
    for X_batch, y_batch in data["train_loader"]:
        print(f"  X_batch shape: {X_batch.shape}")
        print(f"  y_batch shape: {y_batch.shape}")
        print(f"  X_batch device: {X_batch.device}")
        break


def load_from_processed_data(processed_dir: str = None, batch_size: int = None) -> dict:
    """
    Load dữ liệu đã tiền xử lý sẵn từ processed_data/
    
    Dữ liệu này đã qua:
    - Sliding window sequences
    - StandardScaler
    - Train/Val/Test split
    
    Args:
        processed_dir: Thư mục chứa các file .npy
        batch_size: Kích thước batch (default: BATCH_SIZE từ config)
        
    Returns:
        dict: train_loader, val_loader, test_loader, class_weights, config
    """
    import pickle
    
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR
    
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    processed_dir = Path(processed_dir)
    
    print("=" * 60)
    print("LOAD PREPROCESSED DATA")
    print("=" * 60)
    print(f"Directory: {processed_dir}")
    
    # Load arrays
    print("\n[1] Loading numpy arrays...")
    X_train = np.load(processed_dir / "X_train_seq.npy")
    y_train = np.load(processed_dir / "y_train_seq.npy")
    X_val = np.load(processed_dir / "X_val_seq.npy")
    y_val = np.load(processed_dir / "y_val_seq.npy")
    X_test = np.load(processed_dir / "X_test_seq.npy")
    y_test = np.load(processed_dir / "y_test_seq.npy")
    
    print(f"    X_train: {X_train.shape}")
    print(f"    X_val:   {X_val.shape}")
    print(f"    X_test:  {X_test.shape}")
    
    # Class distribution
    print("\n[2] Class distribution:")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        n_normal = (y == 0).sum()
        n_attack = (y == 1).sum()
        total = len(y)
        print(f"    {name}: Normal={n_normal:,} ({n_normal/total*100:.2f}%), Attack={n_attack:,} ({n_attack/total*100:.2f}%)")
    
    # Load config
    print("\n[3] Loading config...")
    config = None
    config_path = processed_dir / "config.pkl"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        print(f"    Time steps: {config.get('time_steps', 20)}")
        print(f"    Features: {config.get('n_features', 15)}")
    
    # Load class weights
    print("\n[4] Loading class weights...")
    class_weights = None
    weights_path = processed_dir / "class_weights.pkl"
    if weights_path.exists():
        with open(weights_path, 'rb') as f:
            class_weights = pickle.load(f)
        print(f"    Normal weight: {class_weights.get(0, 1.0):.4f}")
        print(f"    Attack weight: {class_weights.get(1, 1.0):.4f}")
    
    # Create DataLoaders
    print("\n[5] Creating DataLoaders...")
    print(f"    Batch size: {batch_size}")
    use_cuda = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda
    
    train_dataset = BotIoTDataset(X_train, y_train.astype(np.float32))
    val_dataset = BotIoTDataset(X_val, y_val.astype(np.float32))
    test_dataset = BotIoTDataset(X_test, y_test.astype(np.float32))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETED!")
    print("=" * 60)
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_weights": class_weights,
        "config": config
    }
