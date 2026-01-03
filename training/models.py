"""
=============================================================================
MODELS - Dinh nghia 3 mo hinh Deep Learning
=============================================================================
Muc dich:
- CNN 1D: Baseline - Trich xuat dac trung khong gian
- LSTM: Baseline - Mo hinh hoa chuoi thoi gian
- Hybrid CNN-LSTM: Mo hinh lai (CHUAN IEEE) - Ket hop ca hai

Tat ca models:
- Dung chung input shape: (batch, time_steps, n_features)
- Dung chung output: Sigmoid probability (0-1)
- Toi uu cho GPU

Author: IoT Security Research Team
Date: 2026-01-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import N_FEATURES, TIME_STEPS
except ImportError:
    from config import N_FEATURES, TIME_STEPS


# =============================================================================
# CNN 1D MODEL
# =============================================================================
class CNN1D(nn.Module):
    """
    CNN-1D Model (Baseline)

    Kien truc:
    - 3 Conv1D layers voi BatchNorm va MaxPool
    - Global MaxPooling
    - 2 FC layers

    Dac diem:
    - Trich xuat dac trung cuc bo (local features)
    - Nhanh khi training va inference
    - Khong hoc duoc dependencies dai han
    """

    def __init__(self, n_features: int = N_FEATURES,
                 time_steps: int = TIME_STEPS):
        super(CNN1D, self).__init__()

        self.n_features = n_features
        self.time_steps = time_steps

        # Conv Block 1
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        # Conv Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # Conv Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.4)

        # FC layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, time_steps, n_features)

        Returns:
            logits: (batch,)
        """
        # Transpose: (batch, n_features, time_steps) for Conv1d
        x = x.permute(0, 2, 1)

        # Conv Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Conv Block 3 + Global Pooling
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)  # (batch, 256)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)

        return x.squeeze(-1)


# =============================================================================
# LSTM MODEL
# =============================================================================
class LSTMModel(nn.Module):
    """
    LSTM Model (Baseline)

    Kien truc:
    - 2 LSTM layers (stacked)
    - FC layers voi BatchNorm

    Dac diem:
    - Hoc duoc temporal patterns
    - Nho duoc thong tin dai han (Long-term dependencies)
    - FPR thap nhat tren Bot-IoT dataset
    """

    def __init__(self, n_features: int = N_FEATURES,
                 time_steps: int = TIME_STEPS):
        super(LSTMModel, self).__init__()

        self.n_features = n_features
        self.time_steps = time_steps

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.4)

        # FC layers
        self.fc1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        """
        Args:
            x: (batch, time_steps, n_features)

        Returns:
            logits: (batch,)
        """
        # LSTM Layer 1
        x, _ = self.lstm1(x)  # (batch, time_steps, 128)
        x = self.dropout1(x)

        # LSTM Layer 2
        x, _ = self.lstm2(x)  # (batch, time_steps, 64)
        x = x[:, -1, :]  # Lay output cua timestep cuoi: (batch, 64)
        x = self.dropout2(x)

        # FC layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)

        return x.squeeze(-1)


# =============================================================================
# HYBRID CNN-LSTM MODEL (CHUAN IEEE - KHONG POOLING)
# =============================================================================
class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM Model (CHUAN IEEE PAPERS)

    Kien truc:
    - CNN Block (KHONG co Pooling de giu thong tin temporal)
    - LSTM Block (Hoc temporal patterns tu CNN features)
    - FC Block (Classification)

    TAI SAO KHONG DUNG POOLING?
    - Pooling lam mat thong tin temporal
    - LSTM can do phan giai thoi gian day du
    - Ket qua: FPR giam tu 12.8% xuong 2-3%

    References:
    - IEEE Access: CNN→LSTM for DDoS Detection
    - Scientific Reports (2025): "LSTM uses CNN's output as input"
    """

    def __init__(self, n_features: int = N_FEATURES,
                 time_steps: int = TIME_STEPS):
        super(HybridCNNLSTM, self).__init__()

        self.n_features = n_features
        self.time_steps = time_steps

        # CNN Block (KHONG POOLING!)
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout_cnn = nn.Dropout(0.3)

        # LSTM Block
        self.lstm = nn.LSTM(
            input_size=128,  # Output channels tu CNN
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.dropout_lstm = nn.Dropout(0.3)

        # FC Block
        self.fc1 = nn.Linear(64, 32)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, time_steps, n_features)

        Returns:
            logits: (batch,)
        """
        # CNN Block
        x = x.permute(0, 2, 1)  # (batch, n_features, time_steps)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_cnn(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)
        # Output: (batch, 128, time_steps) - GIU NGUYEN time_steps!

        # LSTM Block
        x = x.permute(0, 2, 1)  # (batch, time_steps, 128)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]  # Hidden state cua layer cuoi: (batch, 64)
        x = self.dropout_lstm(x)

        # FC Block
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc_out(x)

        return x.squeeze(-1)


# =============================================================================
# PARALLEL HYBRID CNN-LSTM MODEL
# =============================================================================
class ParallelHybridCNNLSTM(nn.Module):
    """
    Parallel Hybrid: CNN và LSTM chạy song song, concatenate features
    
    Kiến trúc:
    - LSTM Branch: học temporal patterns trực tiếp từ input
    - CNN Branch: trích xuất spatial features
    - Fusion: concatenate → FC layers
    """
    
    def __init__(self, n_features: int = N_FEATURES,
                 time_steps: int = TIME_STEPS):
        super(ParallelHybridCNNLSTM, self).__init__()
        
        self.n_features = n_features
        self.time_steps = time_steps
        
        # LSTM Branch
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        self.lstm_dropout = nn.Dropout(0.3)
        
        # CNN Branch
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.cnn_dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Fusion layers
        lstm_feat_dim = 64
        cnn_feat_dim = 128 * (time_steps // 4)
        combined_dim = lstm_feat_dim + cnn_feat_dim
        
        self.fc1 = nn.Linear(combined_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc_dropout = nn.Dropout(0.4)
        self.fc_out = nn.Linear(128, 1)
    
    def forward(self, x):
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]  # Last timestep
        lstm_feat = self.lstm_dropout(lstm_feat)
        
        # CNN branch
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.relu(self.bn1(self.conv1(cnn_in)))
        cnn_out = self.pool1(cnn_out)
        cnn_out = self.cnn_dropout(cnn_out)
        
        cnn_out = self.relu(self.bn2(self.conv2(cnn_out)))
        cnn_out = self.pool2(cnn_out)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_feat = cnn_out.flatten(1)
        
        # Concatenate features
        combined = torch.cat((lstm_feat, cnn_feat), dim=1)
        
        # FC layers
        out = self.relu(self.bn_fc(self.fc1(combined)))
        out = self.fc_dropout(out)
        out = self.fc_out(out)
        
        return out.squeeze(-1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_model(model_name: str, n_features: int = N_FEATURES,
              time_steps: int = TIME_STEPS) -> nn.Module:
    """
    Factory function de tao model

    Args:
        model_name: 'CNN', 'LSTM', 'Hybrid', or 'Parallel'
        n_features: So features
        time_steps: So timesteps

    Returns:
        Model instance
    """
    models = {
        'CNN': CNN1D,
        'LSTM': LSTMModel,
        'Hybrid': HybridCNNLSTM,
        'Parallel': ParallelHybridCNNLSTM
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} khong ton tai. "
                        f"Chon tu: {list(models.keys())}")

    return models[model_name](n_features, time_steps)


def count_parameters(model: nn.Module) -> int:
    """Dem so luong parameters cua model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str):
    """In tom tat thong tin model"""
    n_params = count_parameters(model)
    print(f"\n{'=' * 50}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 50}")
    print(f"Parameters: {n_params:,}")
    print(f"{'=' * 50}")
    print(model)


if __name__ == "__main__":
    # Test models
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Dummy input
    batch_size = 32
    x = torch.randn(batch_size, TIME_STEPS, N_FEATURES).to(device)

    for name in ['CNN', 'LSTM', 'Hybrid']:
        model = get_model(name).to(device)
        print_model_summary(model, name)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
            prob = torch.sigmoid(output)

        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{prob.min():.3f}, {prob.max():.3f}]")
        print()
