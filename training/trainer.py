"""
=============================================================================
TRAINER - Class training thong nhat cho tat ca models
=============================================================================
Muc dich:
- Training loop chuan hoa
- Early stopping
- Luu model weights va training history
- Tinh toan metrics trong qua trinh train
- Toi uu cho GPU

Author: IoT Security Research Team
Date: 2026-01-03
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Use new AMP API (torch.amp instead of deprecated torch.cuda.amp)
try:
    from torch.amp import autocast, GradScaler
    AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW_API = False

try:
    from .config import (
        DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
        PATIENCE, MIN_DELTA, OUTPUTS_DIR, LOGS_DIR,
        ACCUMULATION_STEPS
    )
except ImportError:
    from config import (
        DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
        PATIENCE, MIN_DELTA, OUTPUTS_DIR, LOGS_DIR,
        ACCUMULATION_STEPS
    )


# =============================================================================
# FOCAL LOSS - Giải quyết class imbalance tốt hơn BCEWithLogitsLoss
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification

    Focal Loss giảm weight cho các mẫu dễ phân loại (well-classified)
    và tập trung vào các mẫu khó (misclassified).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight cho positive class (Attack). Default: 0.25
        gamma: Focusing parameter. gamma > 0 giảm loss cho well-classified.
               gamma = 0 tương đương BCE. Default: 2.0
        pos_weight: Additional weight cho positive class (từ class_weights)

    References:
        - "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: torch.Tensor = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid)
            targets: Ground truth labels (0 or 1)

        Returns:
            Focal loss value
        """
        # Sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Compute pt (probability of correct class)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Apply pos_weight if provided (for extreme imbalance)
        if self.pos_weight is not None:
            weight = torch.where(targets == 1, self.pos_weight, torch.ones_like(targets))
            alpha_t = alpha_t * weight

        # Focal loss
        focal_weight = (1 - pt) ** self.gamma
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        loss = alpha_t * focal_weight * bce

        return loss.mean()


class EarlyStopping:
    """
    Early Stopping de tranh overfitting

    Dung training neu val_loss khong giam sau `patience` epochs
    """

    def __init__(self, patience: int = PATIENCE,
                 min_delta: float = MIN_DELTA,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Trainer class thong nhat cho tat ca models

    Features:
    - Automatic Mixed Precision (AMP) cho GPU
    - Learning rate scheduling
    - Early stopping
    - Training history logging
    - Model checkpointing
    """

    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 device: torch.device = DEVICE,
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 use_amp: bool = True,
                 class_weights: dict = None,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            model: PyTorch model
            model_name: Ten model ('CNN', 'LSTM', 'Hybrid', 'Parallel')
            device: cuda hoac cpu
            learning_rate: Learning rate
            weight_decay: L2 regularization
            use_amp: Su dung Automatic Mixed Precision (chi GPU)
            class_weights: Dict {0: weight_normal, 1: weight_attack} for imbalanced data
            use_focal_loss: Su dung Focal Loss thay vi BCEWithLogitsLoss
            focal_alpha: Alpha parameter cho Focal Loss (weight cho positive class)
            focal_gamma: Gamma parameter cho Focal Loss (focusing parameter)
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device

        # Loss function selection
        if use_focal_loss:
            # Focal Loss - tốt hơn cho extreme class imbalance
            pos_weight = None
            if class_weights is not None:
                pos_weight = torch.tensor(class_weights[1] / class_weights[0]).to(device)
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                pos_weight=pos_weight
            )
            print(f"[Trainer] Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
            if pos_weight is not None:
                print(f"[Trainer] pos_weight: {pos_weight.item():.6f}")
        elif class_weights is not None:
            # BCEWithLogitsLoss with class weights
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[Trainer] Class weights: Normal={class_weights[0]:.4f}, Attack={class_weights[1]:.4f}")
            print(f"[Trainer] pos_weight: {pos_weight.item():.6f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer (AdamW tot hon Adam cho regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (giam LR khi loss khong giam)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=PATIENCE)

        # AMP (chi dung cho GPU)
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            if AMP_NEW_API:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0

        print(f"[Trainer] Model: {model_name}")
        print(f"[Trainer] Device: {device}")
        print(f"[Trainer] AMP: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"[Trainer] Gradient Accumulation: {ACCUMULATION_STEPS} steps")

    def train_epoch(self, train_loader) -> tuple:
        """
        Train 1 epoch với Gradient Accumulation để tối ưu GPU

        Gradient Accumulation cho phép tích lũy gradients qua nhiều batches
        trước khi update weights, hiệu quả như batch size lớn hơn.

        Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS

        Returns:
            (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation counter
        accumulation_steps = ACCUMULATION_STEPS

        pbar = tqdm(train_loader, desc=f"Training", leave=False)

        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            # Move to device (non_blocking để overlap với compute)
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Forward pass (with AMP if enabled)
            if self.use_amp:
                # Use new API: autocast('cuda') or old API: autocast()
                autocast_ctx = autocast('cuda') if AMP_NEW_API else autocast()
                with autocast_ctx:
                    outputs = self.model(X_batch)
                    # Scale loss by accumulation steps
                    loss = self.criterion(outputs, y_batch) / accumulation_steps

                # Backward pass with scaled gradients (accumulate)
                self.scaler.scale(loss).backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch) / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # Metrics (use unscaled loss for tracking)
            total_loss += loss.item() * accumulation_steps * X_batch.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'acc': f"{correct / total:.4f}"
            })
        
        # Handle remaining gradients (last incomplete accumulation)
        if (batch_idx + 1) % accumulation_steps != 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader) -> tuple:
        """
        Validate model

        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            if self.use_amp:
                autocast_ctx = autocast('cuda') if AMP_NEW_API else autocast()
                with autocast_ctx:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def fit(self,
            train_loader,
            val_loader,
            epochs: int = EPOCHS,
            save_best: bool = True) -> Dict:
        """
        Training loop chinh

        Args:
            train_loader: DataLoader cho training
            val_loader: DataLoader cho validation
            epochs: So epochs
            save_best: Luu model tot nhat

        Returns:
            Training history dict
        """
        print("\n" + "=" * 60)
        print(f"TRAINING: {self.model_name}")
        print("=" * 60)

        total_start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Epoch time
            epoch_time = time.time() - epoch_start

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            # Print progress
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch

                if save_best:
                    self.save_model(f"{self.model_name}_best.pt")
                    print(f"         >> Saved best model (val_loss: {val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n[Early Stopping] No improvement for {PATIENCE} epochs")
                break

        # Training summary
        total_time = time.time() - total_start_time
        self.history['total_time'] = total_time

        print("\n" + "=" * 60)
        print(f"TRAINING HOAN THANH: {self.model_name}")
        print("=" * 60)
        print(f"Best Epoch:    {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc:  {self.best_val_acc:.4f}")
        print(f"Total Time:    {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        print("=" * 60)

        # Save history
        self.save_history()

        return self.history

    def save_model(self, filename: str):
        """Luu model weights"""
        save_path = OUTPUTS_DIR / filename
        torch.save(self.model.state_dict(), save_path)

    def save_history(self):
        """Luu training history ra JSON"""
        history_path = LOGS_DIR / f"{self.model_name}_history.json"

        # Convert numpy types to Python types
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.floating, float)) else v
                    for v in value
                ]
            else:
                history_serializable[key] = float(value) if isinstance(value, (np.floating, float)) else value

        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        print(f"[Trainer] Saved history: {history_path}")

    def load_model(self, filename: str):
        """Load model weights"""
        load_path = OUTPUTS_DIR / filename
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"[Trainer] Loaded model: {load_path}")


def train_model(model_name: str,
                train_loader,
                val_loader,
                epochs: int = EPOCHS) -> Dict:
    """
    Ham tien ich de train mot model

    Args:
        model_name: 'CNN', 'LSTM', hoac 'Hybrid'
        train_loader: DataLoader training
        val_loader: DataLoader validation
        epochs: So epochs

    Returns:
        Training history
    """
    from models import get_model

    # Create model
    model = get_model(model_name)

    # Create trainer
    trainer = Trainer(model, model_name)

    # Train
    history = trainer.fit(train_loader, val_loader, epochs)

    return history


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
