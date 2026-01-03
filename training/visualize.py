#!/usr/bin/env python3
"""
=============================================================================
VISUALIZE - Ve bieu do so sanh cac models
=============================================================================
Muc dich:
- Ve bieu do Loss va Accuracy qua cac epochs
- Ve Confusion Matrix
- Ve ROC Curve
- Ve bieu do so sanh metrics
- Xuat hinh anh cho bao cao

Cach chay:
    python visualize.py
    python visualize.py --output-dir ./figures

Author: IoT Security Research Team
Date: 2026-01-03
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_NAMES, LOGS_DIR, OUTPUTS_DIR

# Check matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib/seaborn not installed. Install with:")
    print("  pip install matplotlib seaborn")


# Colors for each model
MODEL_COLORS = {
    'CNN': '#2196F3',      # Blue
    'LSTM': '#4CAF50',     # Green
    'Hybrid': '#9C27B0'    # Purple
}


def load_history(model_name: str) -> dict:
    """Load training history tu file JSON"""
    history_path = LOGS_DIR / f"{model_name}_history.json"

    if not history_path.exists():
        print(f"WARNING: Khong tim thay {history_path}")
        return None

    with open(history_path, 'r') as f:
        return json.load(f)


def load_evaluation_results() -> dict:
    """Load evaluation results"""
    results_path = LOGS_DIR / "evaluation_results.json"

    if not results_path.exists():
        print(f"WARNING: Khong tim thay {results_path}")
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_curves(output_dir: Path):
    """
    Ve bieu do Loss va Accuracy qua cac epochs

    Tao 2 bieu do:
    1. Training & Validation Loss
    2. Training & Validation Accuracy
    """
    if not HAS_MATPLOTLIB:
        return

    print("\n[1] Ve Training Curves...")

    # Load histories
    histories = {}
    for name in MODEL_NAMES:
        h = load_history(name)
        if h:
            histories[name] = h

    if not histories:
        print("   Khong co du lieu training history!")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1 = axes[0]
    for name, h in histories.items():
        color = MODEL_COLORS[name]
        epochs = range(1, len(h['train_loss']) + 1)

        ax1.plot(epochs, h['train_loss'], '-', color=color, alpha=0.7,
                label=f'{name} Train')
        ax1.plot(epochs, h['val_loss'], '--', color=color,
                label=f'{name} Val')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2 = axes[1]
    for name, h in histories.items():
        color = MODEL_COLORS[name]
        epochs = range(1, len(h['train_acc']) + 1)

        ax2.plot(epochs, h['train_acc'], '-', color=color, alpha=0.7,
                label=f'{name} Train')
        ax2.plot(epochs, h['val_acc'], '--', color=color,
                label=f'{name} Val')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_confusion_matrices(output_dir: Path):
    """Ve Confusion Matrix cho moi model"""
    if not HAS_MATPLOTLIB:
        return

    print("\n[2] Ve Confusion Matrices...")

    results = load_evaluation_results()
    if not results:
        print("   Khong co du lieu evaluation!")
        return

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = metrics['confusion_matrix']
        cm_array = np.array([[cm['tn'], cm['fp']],
                             [cm['fn'], cm['tp']]])

        # Heatmap
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   ax=ax, cbar=False)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}\nAcc: {metrics["accuracy"]:.4f}')

    plt.tight_layout()

    # Save
    save_path = output_dir / "confusion_matrices.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_metrics_comparison(output_dir: Path):
    """Ve bar chart so sanh metrics giua cac models"""
    if not HAS_MATPLOTLIB:
        return

    print("\n[3] Ve Metrics Comparison...")

    results = load_evaluation_results()
    if not results:
        print("   Khong co du lieu evaluation!")
        return

    # Prepare data
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    models = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width,
                     label=model, color=MODEL_COLORS.get(model, 'gray'))

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('So sanh Metrics giua cac Models')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = output_dir / "metrics_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_fpr_fnr_comparison(output_dir: Path):
    """Ve bieu do so sanh FPR va FNR"""
    if not HAS_MATPLOTLIB:
        return

    print("\n[4] Ve FPR/FNR Comparison...")

    results = load_evaluation_results()
    if not results:
        print("   Khong co du lieu evaluation!")
        return

    models = list(results.keys())
    fpr_values = [results[m]['fpr'] * 100 for m in models]  # Convert to %
    fnr_values = [results[m]['fnr'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, fpr_values, width, label='FPR (Bao dong gia)',
                  color='#FF5722')
    bars2 = ax.bar(x + width/2, fnr_values, width, label='FNR (Bo sot tan cong)',
                  color='#F44336')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Rate (%)')
    ax.set_title('So sanh False Positive Rate va False Negative Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = output_dir / "fpr_fnr_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_training_time_comparison(output_dir: Path):
    """Ve bieu do so sanh thoi gian training"""
    if not HAS_MATPLOTLIB:
        return

    print("\n[5] Ve Training Time Comparison...")

    # Load training summary
    summary_path = LOGS_DIR / "training_summary.json"
    if not summary_path.exists():
        print("   Khong co du lieu training summary!")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    models_data = summary.get('models', {})
    if not models_data:
        print("   Khong co du lieu models!")
        return

    models = list(models_data.keys())
    times = [models_data[m]['training_time'] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [MODEL_COLORS.get(m, 'gray') for m in models]
    bars = ax.bar(models, times, color=colors)

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
               f'{t:.1f}s\n({t/60:.1f}m)', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Thoi gian Training cua moi Model')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = output_dir / "training_time_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def create_summary_table(output_dir: Path):
    """Tao bang tom tat dang text/markdown"""
    print("\n[6] Tao Summary Table...")

    results = load_evaluation_results()
    summary_path = LOGS_DIR / "training_summary.json"

    lines = []
    lines.append("# BANG TOM TAT KET QUA")
    lines.append("=" * 80)
    lines.append("")

    # Evaluation metrics
    if results:
        lines.append("## Evaluation Metrics")
        lines.append("-" * 80)
        lines.append(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
                    f"{'F1-Score':>10} {'FPR':>8} {'FNR':>8}")
        lines.append("-" * 80)

        for name, metrics in results.items():
            lines.append(f"{name:<10} "
                        f"{metrics['accuracy']:>10.4f} "
                        f"{metrics['precision']:>10.4f} "
                        f"{metrics['recall']:>10.4f} "
                        f"{metrics['f1_score']:>10.4f} "
                        f"{metrics['fpr'] * 100:>7.2f}% "
                        f"{metrics['fnr'] * 100:>7.2f}%")

        lines.append("-" * 80)
        lines.append("")

    # Training info
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        lines.append("## Training Information")
        lines.append("-" * 80)
        lines.append(f"{'Model':<10} {'Parameters':>12} {'Best Val Loss':>15} "
                    f"{'Best Val Acc':>15} {'Time (s)':>12}")
        lines.append("-" * 80)

        for name, info in summary.get('models', {}).items():
            lines.append(f"{name:<10} "
                        f"{info['n_params']:>12,} "
                        f"{info['best_val_loss']:>15.4f} "
                        f"{info['best_val_acc']:>15.4f} "
                        f"{info['training_time']:>12.1f}")

        lines.append("-" * 80)
        lines.append(f"Total training time: {summary['total_time']:.1f}s "
                    f"({summary['total_time']/60:.1f} minutes)")
        lines.append(f"Device: {summary['device']}")
        lines.append("")

    # Analysis
    if results:
        lines.append("## Phan Tich")
        lines.append("-" * 80)

        best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
        best_fpr = min(results.items(), key=lambda x: x[1]['fpr'])
        best_recall = max(results.items(), key=lambda x: x[1]['recall'])

        lines.append(f"- Model co Accuracy cao nhat:  {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
        lines.append(f"- Model co F1-Score cao nhat:  {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        lines.append(f"- Model co FPR thap nhat:      {best_fpr[0]} ({best_fpr[1]['fpr'] * 100:.2f}%)")
        lines.append(f"- Model co Recall cao nhat:    {best_recall[0]} ({best_recall[1]['recall']:.4f})")
        lines.append("")

    # Save
    content = "\n".join(lines)
    save_path = output_dir / "summary_table.txt"
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"   Saved: {save_path}")
    print("\n" + content)


def main():
    parser = argparse.ArgumentParser(
        description="Ve bieu do so sanh cac Deep Learning Models"
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Thu muc luu hinh anh (default: training/outputs/figures)'
    )

    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUTS_DIR / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VE BIEU DO SO SANH MODELS")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print("=" * 60)

    if not HAS_MATPLOTLIB:
        print("\nERROR: Can cai dat matplotlib va seaborn!")
        print("  pip install matplotlib seaborn")
        sys.exit(1)

    # Generate all plots
    plot_training_curves(output_dir)
    plot_confusion_matrices(output_dir)
    plot_metrics_comparison(output_dir)
    plot_fpr_fnr_comparison(output_dir)
    plot_training_time_comparison(output_dir)
    create_summary_table(output_dir)

    print("\n" + "=" * 60)
    print("HOAN THANH!")
    print("=" * 60)
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
