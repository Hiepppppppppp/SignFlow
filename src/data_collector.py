from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import cv2
import numpy as np

try:
    from .hand_tracker import HandTracker
except ImportError:
    from hand_tracker import HandTracker

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "landmarks"
SEQ_DIR = ROOT / "data" / "sequences"
DATASET_DIR = ROOT / "dataset" / "asl_alphabet_train" / "asl_alphabet_train"


def collect_images(label: str, samples: int, camera: int):
    """Collect 200x200 cropped/padded hand images directly into dataset folder."""
    dest_dir = DATASET_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest_dir.glob("*.jpg")))

    tracker = HandTracker()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"Không mở được webcam index={camera}")

    count = 0
    print(f"Bắt đầu thu thập ảnh cho cử chỉ '{label}' vào thư mục: {dest_dir}")
    print("Đưa tay vào khung hình. Nhấn S để lưu mẫu (hoặc giữ phím Space để tự động lưu liên tục). Nhấn Q để dừng.")

    try:
        while count < samples:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame, draw=True)

            status_text = f"Cử chỉ: {label} | Đã lưu: {count}/{samples} | Nhấn S: Lưu, Q: Thoát"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("SignFlow Data Collector - Image Mode", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if hand.detected and (key == ord("s") or key == 32):  # 's' or Space
                x1, y1, x2, y2 = hand.bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    resized = cv2.resize(crop, (200, 200))
                    filename = dest_dir / f"{label}_{existing + count + 1}.jpg"
                    cv2.imwrite(str(filename), resized)
                    count += 1
                    time.sleep(0.06)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"Đã lưu thành công {count} ảnh cho cử chỉ '{label}' vào: {dest_dir}")


def collect_static(label: str, samples: int, camera: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "landmarks.csv"
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"Không mở được webcam index={camera}")
    exists = csv_path.exists()
    count = 0
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["label"] + [f"f{i}" for i in range(63)])
            print(f"Thu thập landmarks cho cử chỉ '{label}'. Nhấn S để lưu, Q để dừng.")
            while count < samples:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                hand = tracker.process(frame, draw=True)
                cv2.putText(frame, f"Label: {label} | Saved: {count}/{samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("SignFlow Landmark Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if hand.detected and (key == ord("s") or key == 32):
                    writer.writerow([label] + hand.landmarks.reshape(-1).tolist())
                    count += 1
                    time.sleep(0.08)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    print(f"Đã lưu {count} mẫu vào {csv_path}")


def collect_sequence(label: str, frames: int, samples: int, camera: int):
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"Không mở được webcam index={camera}")
    saved = 0
    try:
        print("Nhấn S để bắt đầu ghi một chuỗi. Giữ tay trong khung hình. Q để thoát.")
        while saved < samples:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame, draw=True)
            cv2.putText(frame, f"{label} | sequences: {saved}/{samples} | S=start",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("SignFlow Sequence Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                seq, deadline = [], time.time() + 12
                while len(seq) < frames and time.time() < deadline:
                    ok2, fr = cap.read()
                    if not ok2:
                        continue
                    fr = cv2.flip(fr, 1)
                    h = tracker.process(fr, draw=True)
                    if h.detected:
                        seq.append(h.landmarks.reshape(-1))
                    cv2.putText(fr, f"Recording {len(seq)}/{frames}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("SignFlow Sequence Collector", fr)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return
                if len(seq) == frames:
                    path = SEQ_DIR / f"{label}_{int(time.time()*1000)}.npy"
                    np.save(path, np.asarray(seq, dtype=np.float32))
                    saved += 1
                    print("Saved:", path.name)
                else:
                    print("Chuỗi không đủ frame, bỏ mẫu.")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ thu thập dữ liệu cử chỉ tay mới")
    parser.add_argument("--mode", choices=["images", "static", "sequence"], default="images",
                        help="images: lưu ảnh vào dataset, static: lưu landmarks vào csv, sequence: lưu chuỗi npy")
    parser.add_argument("--label", required=True, help="Tên nhãn cử chỉ, ví dụ: Like, Dislike, Heart, OK, v.v.")
    parser.add_argument("--samples", type=int, default=100, help="Số lượng mẫu cần thu thập (mặc định 100)")
    parser.add_argument("--frames", type=int, default=32, help="Số frame cho mỗi sequence (chế độ sequence)")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số webcam")
    args = parser.parse_args()

    if args.mode == "images":
        collect_images(args.label, args.samples, args.camera)
    elif args.mode == "static":
        collect_static(args.label, args.samples, args.camera)
    else:
        collect_sequence(args.label, args.frames, args.samples, args.camera)
