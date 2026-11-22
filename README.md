# SignFlow - Hệ thống nhận diện cử chỉ tay (Hand Gesture Recognition)

Hệ thống nhận diện cử chỉ tay thời gian thực sử dụng **21 MediaPipe hand landmarks** và mô hình phân loại nơ-ron đa tầng (MLP Classifier) kết hợp bộ phân tích hình học cử chỉ chính xác cao.

Hệ thống hiển thị **đầy đủ tên cử chỉ rõ ràng (tiếng Anh & tiếng Việt)** thay vì chỉ hiển thị ký tự đơn lẻ (như j, k, s,...), đồng thời hỗ trợ nhận diện các cử chỉ phổ biến như **Like (Thích / Thumbs Up)**, **Peace / Victory (Chữ V)**, **Fist (Nắm đấm / Chữ S)**, **Open Palm (Bàn tay mở / Space)**, **Call Me / Shaka (Chữ Y)**, **L-Sign (Chữ L)**, **OK Sign**, v.v.

---

## 1. Cấu trúc thư mục Dataset

```text
dataset/asl_alphabet_train/asl_alphabet_train/
  ├── Like/           # Cử chỉ Like (Ngón cái giơ lên)
  ├── J/              # Cử chỉ ngón út (Chữ J)
  ├── K/              # Cử chỉ chữ K
  ├── L/              # Cử chỉ chữ L (Góc vuông)
  ├── M/              # Ba ngón gập
  ├── N/              # Hai ngón gập
  ├── O/              # Khum tròn chữ O
  ├── P/              # Chữ P chúc xuống
  ├── Q/              # Chữ Q chúc xuống
  ├── R/              # Bắt chéo ngón (Chúc may mắn)
  ├── S/              # Nắm đấm (Chữ S)
  ├── space/          # Bàn tay mở / Stop (Space)
  ├── T/              # Ngón cái kẹp giữa
  ├── U/              # Hai ngón chụm thẳng
  ├── V/              # Hai ngón chữ V (Peace / Victory)
  ├── W/              # Ba ngón xòe (Số 3 / Chữ W)
  ├── X/              # Ngón trỏ móc (Hook)
  ├── Y/              # Ngón cái + út (Call Me / Shaka)
  └── Z/              # Chỉ ngón trỏ (Point / Chữ Z)
```

Tên chi tiết và ý nghĩa của tất cả cử chỉ được cấu hình tập trung trong file:
`models/gesture_names.json` và mã nguồn `src/gestures.py`.

---

## 2. Cài đặt môi trường

Từ thư mục dự án (`SignFlow-main`):

```bash
pip install -r requirements.txt
```

---

## 3. Chạy nhận diện qua Webcam

Chạy chương trình nhận diện thời gian thực:

```bash
python -m src.webcam
```

### Các phím tắt tiện ích khi bật Webcam:
- **`Q` hoặc `Esc`**: Thoát chương trình webcam.
- **`H`**: Bật/Tắt bảng danh sách hướng dẫn các cử chỉ tay ngay trên màn hình camera.
- **`M`**: Bật/Tắt chế độ lật hình camera (Lật ngang).
- Tuỳ chỉnh ngưỡng nhận diện: `python -m src.webcam --threshold 0.70`

---

## 4. Thu thập thêm cử chỉ mới vào Dataset

Để thu thập thêm ảnh cho một cử chỉ mới (ví dụ: `Heart`, `Dislike`, `OK`,...):

```bash
# Lưu ảnh trực tiếp vào dataset:
python -m src.data_collector --mode images --label Like --samples 100
```
- Khi mở webcam, đưa tay vào khung hình và nhấn phím **`S`** (hoặc giữ phím Space) để lưu ảnh tay.

---

## 5. Trích xuất Landmark & Huấn luyện lại mô hình

### Bước 1: Trích xuất 21 điểm landmarks từ thư mục ảnh dataset:
```bash
python -m src.build_dataset --dataset "dataset/asl_alphabet_train/asl_alphabet_train" --output "data/landmarks.npz"
```

### Bước 2: Huấn luyện mô hình phân loại:
```bash
python -m src.train --data "data/landmarks.npz" --output "models/landmark_model.pth" --epochs 50
```
Mô hình sẽ tự động cập nhật danh sách lớp và lưu vào `models/landmark_model.pth` cùng `models/landmark_classes.json`.
