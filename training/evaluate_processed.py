#!/usr/bin/env python3
"""
=============================================================================
EVALUATE PROCESSED - Đánh giá models với dữ liệu đã xử lý
=============================================================================
Load test set từ processed_data/ và đánh giá tất cả models

Cách chạy:
    python evaluate_processed.py                  # Đánh giá tất cả
    python evaluate_processed.py --models CNN     # Đánh giá một model

Author: IoT Security Research Team
Date: 2026-01-03
"""

import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, MODEL_NAMES_ALL, OUTPUTS_DIR, LOGS_DIR, PROCESSED_DATA_DIR
from data_loader import load_from_processed_data
from models import get_model


class ModelEvaluator:
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.results = {}

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, test_loader) -> tuple:
        model.eval()
        all_probs = []
        all_labels = []

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob > 0.5).astype(int)

        return y_true, y_pred, y_prob

    def calculate_metrics(self, y_true, y_pred, y_prob) -> dict:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, len(y_true)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'total_samples': len(y_true),
            'total_attacks': int((y_true == 1).sum()),
            'total_normal': int((y_true == 0).sum())
        }

    def evaluate_model(self, model_name: str, test_loader, model_dir: Path = OUTPUTS_DIR) -> dict:
        print(f"\n{'=' * 50}")
        print(f"ĐÁNH GIÁ MODEL: {model_name}")
        print(f"{'=' * 50}")

        # Try multiple model file patterns
        patterns = [
            f"{model_name}_best.pt",
            f"{model_name}_CNN_LSTM_best.pt" if model_name == "Hybrid" else None,
            f"Parallel_Hybrid_best.pt" if model_name == "Parallel" else None
        ]
        
        model_path = None
        for p in patterns:
            if p and (model_dir / p).exists():
                model_path = model_dir / p
                break
        
        if model_path is None:
            print(f"WARNING: Không tìm thấy model {model_name} trong {model_dir}")
            return None

        model = get_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        print(f"Loaded: {model_path}")

        start_time = time.time()
        y_true, y_pred, y_prob = self.predict(model, test_loader)
        inference_time = time.time() - start_time

        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        metrics['inference_time'] = inference_time
        metrics['inference_time_per_sample'] = inference_time / len(y_true) * 1000

        self._print_metrics(model_name, metrics)
        self.results[model_name] = {'metrics': metrics}

        # Save classification report
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'])
        report_path = LOGS_DIR / f"{model_name}_classification_report_processed.txt"
        with open(report_path, 'w') as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        print(f"Report saved: {report_path}")

        return metrics

    def _print_metrics(self, model_name: str, metrics: dict):
        print(f"\n--- {model_name} Metrics ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nFalse Positive Rate (FPR): {metrics['fpr']:.4f}")
        print(f"False Negative Rate (FNR): {metrics['fnr']:.4f}")

        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal  Attack")
        print(f"Actual Normal    {cm['tn']:6d}  {cm['fp']:6d}")
        print(f"       Attack    {cm['fn']:6d}  {cm['tp']:6d}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate models with processed data')
    parser.add_argument('--models', nargs='+', default=['CNN', 'LSTM', 'Hybrid'],
                        choices=MODEL_NAMES_ALL, help='Models to evaluate')
    parser.add_argument('--data-dir', type=str, default=str(PROCESSED_DATA_DIR),
                        help='Directory containing processed data')
    parser.add_argument('--model-dir', type=str, default=str(OUTPUTS_DIR),
                        help='Directory containing model weights')
    args = parser.parse_args()

    print("=" * 60)
    print("ĐÁNH GIÁ MODELS VỚI PROCESSED DATA")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    data = load_from_processed_data(args.data_dir)
    test_loader = data['test_loader']

    # Evaluate
    evaluator = ModelEvaluator(DEVICE)
    all_results = {}

    for model_name in args.models:
        metrics = evaluator.evaluate_model(model_name, test_loader, Path(args.model_dir))
        if metrics:
            all_results[model_name] = metrics

    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SO SÁNH CÁC MODELS")
        print("=" * 60)
        print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>8} {'FNR':>8}")
        print("-" * 70)
        for name, m in all_results.items():
            print(f"{name:<10} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} "
                  f"{m['fpr']:>8.4f} {m['fnr']:>8.4f}")
        print("-" * 70)

    # Save results
    results_path = LOGS_DIR / "evaluation_results_processed.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
