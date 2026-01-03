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

## 1) Chuẩn Bị Dữ Liệu
- Nếu đã có file merge đầy đủ (có cột `attack`), ví dụ: `E:/Bot_IOT_Dataset/Dataset/Dataset/Entire Dataset/UNSW_2018_IoT_Botnet_Entire_Merged.csv`.
- Tuỳ chọn (khuyến nghị): Tạo tập cân bằng nhỏ để train/đánh giá có ý nghĩa hơn:
  ```powershell
  python training/create_balanced_subset.py --source "E:/.../UNSW_2018_IoT_Botnet_Entire_Merged.csv" --normal_target 50000 --attack_target 50000 --chunksize 200000
  ```
  Kết quả lưu trong `training/outputs/balanced/` (train/val/test CSV + class_weights).

## 2) Chạy Training
- Vào thư mục training:
  ```powershell
  cd training
  ```
- Train tất cả models trên file dữ liệu bạn chọn (ưu tiên file cân bằng nếu đã tạo):
  ```powershell
  python train_all.py --data "E:/.../UNSW_2018_IoT_Botnet_Entire_Merged.csv" --epochs 30
  ```
  Hoặc train riêng từng model: thêm `--models CNN` hoặc `--models LSTM` hoặc `--models Hybrid`.
- Kết quả chính nằm ở `training/outputs/` (weights *_best.pt, scaler_standard.pkl, X_test.npy, y_test.npy) và `training/logs/` (history, reports).

## 3) Đánh Giá
- Sau khi train xong (vẫn ở thư mục training):
  ```powershell
  python evaluate.py
  ```
- Kết quả: `training/logs/evaluation_results.json` và các `*_classification_report.txt` trong `training/logs/`.

## 4) Chuẩn Bị Cho Web Demo
- Copy các file sau vào `backend/models/`:
  - `training/outputs/CNN_best.pt`
  - `training/outputs/LSTM_best.pt`
  - `training/outputs/Hybrid_best.pt` (hoặc tên tương ứng)
  - `training/outputs/scaler_standard.pkl`
- Tạo dữ liệu demo nhỏ (nếu chưa có `data/demo_test.csv`):
  ```powershell
  cd ..   # quay lại root
  python prepare_demo_data.py
  ```
  (Tạo 500 Normal + 500 Attack mẫu demo.)

## 5) Chạy Web
- Ở thư mục gốc (đảm bảo đã kích hoạt venv):
  ```powershell
  python app.py
  ```
- Mở trình duyệt: http://localhost:5000
- Nút Start/Stop điều khiển replay; xem realtime dashboard.

## 6) Ghi Chú Nhanh
- Nếu muốn ép chạy CPU: chỉnh `self.device = torch.device('cpu')` trong `backend/replay_detector.py`.
- Nếu thiếu bộ nhớ khi merge dữ liệu lớn: dùng `merge_entire_streaming.py` (đọc chunk, ghi thẳng ra file).
- Để đánh giá có ý nghĩa (tránh FPR=1.0): dùng tập cân bằng hoặc class weights từ `training/outputs/balanced/class_weights.json`.
