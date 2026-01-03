# Hướng Dẫn Chạy Nhanh (Training + Web)

## 0) Chuẩn Bị Môi Trường
- Python 3.8+
- Tạo và kích hoạt venv (Windows):
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
- Cài PyTorch (chọn GPU nếu có CUDA 11.8):
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- Cài dependencies còn lại:
  ```powershell
  pip install -r requirements.txt
  ```

## 1) Chạy Training (Khuyến nghị: dùng dữ liệu đã xử lý)

Dữ liệu đã được tiền xử lý sẵn trong `processed_data/` (sequences, scaled, class weights).

```powershell
cd training

# Train tất cả models (CNN, LSTM, Hybrid) với class weights
python train_processed.py

# Train model cụ thể
python train_processed.py --models LSTM

# Train Parallel Hybrid
python train_processed.py --models Parallel

# Tùy chỉnh epochs
python train_processed.py --epochs 30 --models CNN
```

### Hoặc: Training từ CSV gốc (nếu có file CSV)
```powershell
cd training
python train_all.py --data /path/to/botiot.csv --epochs 50
```

## 2) Đánh Giá Models

```powershell
cd training

# Đánh giá với processed test data
python evaluate_processed.py

# Đánh giá model cụ thể
python evaluate_processed.py --models CNN LSTM

# Dùng models từ backend/models
python evaluate_processed.py --model-dir ../backend/models
```

Kết quả: `training/logs/evaluation_results_processed.json` và `*_classification_report_processed.txt`.

## 3) Chuẩn Bị Cho Web Demo

- Các model weights đã có sẵn trong `backend/models/`:
  - `CNN_best.pt`, `LSTM_best.pt`, `Hybrid_CNN_LSTM_best.pt`, `Parallel_Hybrid_best.pt`
  - `scaler_standard.pkl`

- Nếu vừa train xong, copy từ `training/outputs/` sang `backend/models/`.

- Tạo dữ liệu demo nhỏ (nếu chưa có `data/demo_test.csv`):
  ```powershell
  cd ..   # quay lại root
  python prepare_demo_data.py
  ```

## 4) Chạy Web Demo

```powershell
# Ở thư mục gốc (đảm bảo đã kích hoạt venv)
python app.py
```

Mở trình duyệt: http://localhost:5000

Nhấn "Start Replay" để xem dashboard real-time.

## 5) Ghi Chú
- Nếu muốn ép chạy CPU: chỉnh `self.device = torch.device('cpu')` trong `backend/replay_detector.py`.
- Dữ liệu rất mất cân bằng (99.6% Attack). Dùng `train_processed.py` với class weights để cải thiện.
- Models trong `backend/models/` dùng cho web demo; models trong `training/outputs/` là kết quả training mới.
