#!/usr/bin/env python3
"""
=============================================================================
EVALUATE - Danh gia va so sanh 3 models
=============================================================================
Muc dich:
- Load test set CHUNG cho tat ca models
- Tinh toan metrics: Accuracy, Precision, Recall, F1-Score
- Tao Confusion Matrix
- So sanh hieu nang giua cac models
- Xuat bao cao chi tiet

QUAN TRONG: Dam bao tat ca thanh vien dung cung test set!

Cach chay:
    python evaluate.py
    python evaluate.py --models CNN LSTM Hybrid

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
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, MODEL_NAMES, OUTPUTS_DIR, LOGS_DIR
from data_loader import load_test_set_only
from models import get_model


class ModelEvaluator:
    """
    Class de danh gia models tren test set

    Cung cap:
    - Confusion Matrix
    - Classification Report (Accuracy, Precision, Recall, F1)
    - ROC-AUC Score
    - False Positive Rate (FPR)
    - False Negative Rate (FNR)
    """

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.results = {}

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, test_loader) -> tuple:
        """
        Chay inference tren test set

        Returns:
            y_true, y_pred, y_prob
        """
        model.eval()
        all_probs = []
        all_labels = []

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)

            # Forward pass
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob > 0.5).astype(int)

        return y_true, y_pred, y_prob

    def calculate_metrics(self, y_true, y_pred, y_prob) -> dict:
        """
        Tinh toan tat ca metrics

        Returns:
            dict chua cac metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = 0.0

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Tinh FPR va FNR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,  # Ty le bao dong gia
            'fnr': fnr,  # Ty le bo sot tan cong
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'total_samples': len(y_true),
            'total_attacks': int((y_true == 1).sum()),
            'total_normal': int((y_true == 0).sum())
        }

    def evaluate_model(self, model_name: str, test_loader) -> dict:
        """
        Danh gia mot model

        Args:
            model_name: 'CNN', 'LSTM', hoac 'Hybrid'
            test_loader: DataLoader cho test set

        Returns:
            dict chua metrics
        """
        print(f"\n{'=' * 50}")
        print(f"DANH GIA MODEL: {model_name}")
        print(f"{'=' * 50}")

        # Load model
        model = get_model(model_name)
        model_path = OUTPUTS_DIR / f"{model_name}_best.pt"

        if not model_path.exists():
            print(f"WARNING: Khong tim thay {model_path}")
            return None

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        print(f"Loaded: {model_path}")

        # Predict
        start_time = time.time()
        y_true, y_pred, y_prob = self.predict(model, test_loader)
        inference_time = time.time() - start_time

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        metrics['inference_time'] = inference_time
        metrics['inference_time_per_sample'] = inference_time / len(y_true) * 1000  # ms

        # Print results
        self._print_metrics(model_name, metrics)

        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        return metrics

    def _print_metrics(self, model_name: str, metrics: dict):
        """In metrics ra console"""
        print(f"\n--- {model_name} Metrics ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nFalse Positive Rate (FPR): {metrics['fpr']:.4f} ({metrics['fpr'] * 100:.2f}%)")
        print(f"False Negative Rate (FNR): {metrics['fnr']:.4f} ({metrics['fnr'] * 100:.2f}%)")

        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal  Attack")
        print(f"Actual Normal    {cm['tn']:6d}  {cm['fp']:6d}")
        print(f"       Attack    {cm['fn']:6d}  {cm['tp']:6d}")

        print(f"\nInference time: {metrics['inference_time']:.3f}s "
              f"({metrics['inference_time_per_sample']:.3f}ms/sample)")

    def evaluate_all(self, test_loader, models: list = None) -> dict:
        """
        Danh gia tat ca models

        Args:
            test_loader: DataLoader cho test set
            models: List models can danh gia (default: tat ca)

        Returns:
            dict chua metrics cua tat ca models
        """
        if models is None:
            models = MODEL_NAMES

        all_metrics = {}

        for model_name in models:
            metrics = self.evaluate_model(model_name, test_loader)
            if metrics:
                all_metrics[model_name] = metrics

        # Print comparison
        self._print_comparison(all_metrics)

        # Save results
        self._save_results(all_metrics)

        return all_metrics

    def _print_comparison(self, all_metrics: dict):
        """In bang so sanh cac models"""
        print("\n" + "=" * 80)
        print("SO SANH CAC MODELS")
        print("=" * 80)

        # Header
        print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
              f"{'F1-Score':>10} {'FPR':>8} {'FNR':>8} {'Time(ms)':>10}")
        print("-" * 80)

        # Data
        for name, metrics in all_metrics.items():
            print(f"{name:<10} "
                  f"{metrics['accuracy']:>10.4f} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f} "
                  f"{metrics['fpr']:>8.4f} "
                  f"{metrics['fnr']:>8.4f} "
                  f"{metrics['inference_time_per_sample']:>10.3f}")

        print("-" * 80)

        # Tim model tot nhat theo tung metric
        if all_metrics:
            best_acc = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
            best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
            best_fpr = min(all_metrics.items(), key=lambda x: x[1]['fpr'])
            best_recall = max(all_metrics.items(), key=lambda x: x[1]['recall'])

            print(f"\nBest Accuracy:  {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
            print(f"Best F1-Score:  {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
            print(f"Lowest FPR:     {best_fpr[0]} ({best_fpr[1]['fpr']:.4f})")
            print(f"Best Recall:    {best_recall[0]} ({best_recall[1]['recall']:.4f})")

        print("=" * 80)

    def _save_results(self, all_metrics: dict):
        """Luu ket qua ra file"""
        # Convert numpy types to Python types
        save_metrics = {}
        for model_name, metrics in all_metrics.items():
            save_metrics[model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
            }

        results_path = LOGS_DIR / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        print(f"\nSaved results: {results_path}")

        # Luu classification report chi tiet
        for model_name in self.results:
            y_true = self.results[model_name]['y_true']
            y_pred = self.results[model_name]['y_pred']

            report = classification_report(
                y_true, y_pred,
                target_names=['Normal', 'Attack'],
                digits=4
            )

            report_path = LOGS_DIR / f"{model_name}_classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Classification Report: {model_name}\n")
                f.write("=" * 50 + "\n")
                f.write(report)
            print(f"Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Danh gia va so sanh 3 Deep Learning Models"
    )

    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help=f'Models can danh gia (default: {MODEL_NAMES})'
    )

    parser.add_argument(
        '--test-dir', '-t',
        type=str,
        default=None,
        help='Thu muc chua X_test.npy va y_test.npy'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DANH GIA MODELS")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load test set
    try:
        X_test, y_test, test_loader = load_test_set_only(args.test_dir)
    except FileNotFoundError as e:
        print(f"\nERROR: Khong tim thay test set!")
        print("Hay chay train_all.py truoc de tao test set.")
        print(f"Hoac chi dinh duong dan: --test-dir path/to/test/")
        sys.exit(1)

    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(test_loader, args.models)

    print("\n" + "=" * 60)
    print("HOAN THANH DANH GIA!")
    print("=" * 60)
    print(f"Results: {LOGS_DIR}/evaluation_results.json")
    print(f"Reports: {LOGS_DIR}/*_classification_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
