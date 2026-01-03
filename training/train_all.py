#!/usr/bin/env python3
"""
=============================================================================
TRAIN ALL MODELS - Script chinh de train 3 models
=============================================================================
Muc dich:
- Train 3 models (CNN, LSTM, Hybrid) tren cung dataset
- Su dung cung cach danh gia
- Luu model weights, history, va thoi gian training
- Toi uu cho GPU

Cach chay:
    python train_all.py --data path/to/botiot.csv
    python train_all.py --data path/to/botiot.csv --models CNN LSTM
    python train_all.py --data path/to/botiot.csv --epochs 100

Author: IoT Security Research Team
Date: 2026-01-03
"""

import argparse
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, EPOCHS, MODEL_NAMES, OUTPUTS_DIR, LOGS_DIR,
    print_config
)
from data_loader import load_and_preprocess_data
from models import get_model, count_parameters, print_model_summary
from trainer import Trainer


def train_single_model(model_name: str,
                       train_loader,
                       val_loader,
                       epochs: int) -> dict:
    """
    Train mot model

    Returns:
        dict chua history va thong tin model
    """
    print(f"\n{'#' * 60}")
    print(f"# TRAINING MODEL: {model_name}")
    print(f"{'#' * 60}")

    # Create model
    model = get_model(model_name)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Create trainer
    trainer = Trainer(model, model_name)

    # Train
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, epochs)
    training_time = time.time() - start_time

    # Return results
    return {
        'model_name': model_name,
        'n_params': n_params,
        'history': history,
        'training_time': training_time,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch
    }


def train_all_models(data_path: str,
                     models_to_train: list = None,
                     epochs: int = EPOCHS) -> dict:
    """
    Train tat ca models

    Args:
        data_path: Duong dan den file CSV
        models_to_train: List cac model can train (default: tat ca)
        epochs: So epochs

    Returns:
        dict chua ket qua cua tat ca models
    """
    if models_to_train is None:
        models_to_train = MODEL_NAMES

    print("\n" + "=" * 60)
    print("TRAIN ALL MODELS")
    print("=" * 60)
    print(f"Models: {models_to_train}")
    print(f"Epochs: {epochs}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    data = load_and_preprocess_data(data_path)
    train_loader = data['train_loader']
    val_loader = data['val_loader']

    # Train each model
    results = {}
    total_start = time.time()

    for model_name in models_to_train:
        result = train_single_model(model_name, train_loader, val_loader, epochs)
        results[model_name] = result

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("TOM TAT KET QUA TRAINING")
    print("=" * 60)
    print(f"{'Model':<10} {'Params':>12} {'Best Loss':>12} {'Best Acc':>12} {'Time (s)':>12}")
    print("-" * 60)

    for name, result in results.items():
        print(f"{name:<10} {result['n_params']:>12,} "
              f"{result['best_val_loss']:>12.4f} "
              f"{result['best_val_acc']:>12.4f} "
              f"{result['training_time']:>12.1f}")

    print("-" * 60)
    print(f"{'TOTAL':<10} {'':<12} {'':<12} {'':<12} {total_time:>12.1f}")
    print("=" * 60)

    # Save summary
    summary = {
        'models': {},
        'total_time': total_time,
        'device': str(DEVICE),
        'epochs': epochs
    }

    for name, result in results.items():
        summary['models'][name] = {
            'n_params': result['n_params'],
            'best_val_loss': result['best_val_loss'],
            'best_val_acc': result['best_val_acc'],
            'best_epoch': result['best_epoch'],
            'training_time': result['training_time']
        }

    summary_path = LOGS_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train 3 Deep Learning Models cho Bot-IoT DDoS Detection"
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Duong dan den file CSV Bot-IoT dataset'
    )

    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help=f'Models can train (default: {MODEL_NAMES})'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=EPOCHS,
        help=f'So epochs (default: {EPOCHS})'
    )

    args = parser.parse_args()

    # Print config
    print_config()

    # Check data file
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: File khong ton tai: {data_path}")
        sys.exit(1)

    # Train
    results = train_all_models(
        str(data_path),
        args.models,
        args.epochs
    )

    print("\n" + "=" * 60)
    print("HOAN THANH!")
    print("=" * 60)
    print(f"Model weights: {OUTPUTS_DIR}/")
    print(f"Training logs: {LOGS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
