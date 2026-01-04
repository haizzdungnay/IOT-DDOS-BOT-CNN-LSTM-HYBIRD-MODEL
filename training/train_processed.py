#!/usr/bin/env python3
"""
=============================================================================
TRAIN WITH PROCESSED DATA - Train models từ dữ liệu đã tiền xử lý
=============================================================================
Sử dụng dữ liệu từ processed_data/ (đã có sẵn sequences, scaled, split)
Hỗ trợ class weights cho dữ liệu mất cân bằng

Cách chạy:
    python train_processed.py                    # Train tất cả models
    python train_processed.py --models CNN LSTM  # Train một số models
    python train_processed.py --epochs 30        # Thay đổi epochs
    python train_processed.py --no-weights       # Không dùng class weights

Author: IoT Security Research Team
Date: 2026-01-03
"""

import argparse
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAMES_ALL, OUTPUTS_DIR, LOGS_DIR, PROCESSED_DATA_DIR,
    GRADIENT_ACCUMULATION_STEPS
)
from data_loader import load_from_processed_data
from models import get_model, count_parameters
from trainer import Trainer


def train_single_model(model_name: str,
                       train_loader,
                       val_loader,
                       epochs: int,
                       learning_rate: float = 0.001,
                       class_weights: dict = None,
                       accumulation_steps: int = 2) -> dict:
    """Train một model với GPU optimization"""
    print(f"\n{'#' * 60}")
    print(f"# TRAINING MODEL: {model_name}")
    print(f"{'#' * 60}")

    # Create model
    model = get_model(model_name)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient Accumulation: {accumulation_steps} steps")
    print(f"Effective batch size: {train_loader.batch_size * accumulation_steps}")

    # Create trainer with class weights and learning rate
    trainer = Trainer(model, model_name, learning_rate=learning_rate, class_weights=class_weights)

    # Train với accumulation steps
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, epochs, 
                          accumulation_steps=accumulation_steps,
                          compile_model=False)  # Tắt compile trên Windows
    training_time = time.time() - start_time

    return {
        'model_name': model_name,
        'n_params': n_params,
        'history': history,
        'training_time': training_time,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch
    }


def main():
    parser = argparse.ArgumentParser(description='Train models with processed data - GPU Optimized')
    parser.add_argument('--models', nargs='+', default=['CNN', 'LSTM', 'Hybrid'],
                        choices=MODEL_NAMES_ALL,
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--data-dir', type=str, default=str(PROCESSED_DATA_DIR),
                        help='Directory containing processed data')
    parser.add_argument('--accum-steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help='Gradient accumulation steps')
    parser.add_argument('--no-weights', action='store_true',
                        help='Disable class weights (not recommended for imbalanced data)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRAIN WITH PROCESSED DATA - GPU OPTIMIZED")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient Accumulation: {args.accum_steps}")
    print(f"Effective batch: {args.batch_size * args.accum_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Data dir: {args.data_dir}")
    print(f"Device: {DEVICE}")
    print(f"Use class weights: {not args.no_weights}")
    print("=" * 60)

    # Load processed data
    data = load_from_processed_data(args.data_dir, batch_size=args.batch_size)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    class_weights = None if args.no_weights else data.get('class_weights')

    # Train each model
    results = {}
    total_start = time.time()

    for model_name in args.models:
        result = train_single_model(
            model_name, train_loader, val_loader,
            args.epochs, args.lr, class_weights,
            accumulation_steps=args.accum_steps
        )
        results[model_name] = result

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
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
        'epochs': args.epochs,
        'class_weights_used': not args.no_weights
    }

    for name, result in results.items():
        summary['models'][name] = {
            'n_params': result['n_params'],
            'best_val_loss': result['best_val_loss'],
            'best_val_acc': result['best_val_acc'],
            'best_epoch': result['best_epoch'],
            'training_time': result['training_time']
        }

    summary_path = LOGS_DIR / "training_summary_processed.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Model weights: {OUTPUTS_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
