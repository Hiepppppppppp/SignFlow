"""Real-time webcam inference with full gesture catalog (A-Z, Like, etc.) and flip controls."""
from __future__ import annotations
import argparse
from collections import deque, Counter
from pathlib import Path
import cv2
import numpy as np

try:
    from .config import DEFAULT_MODEL_PATH
    from .landmarks import HandLandmarkExtractor
    from .predictor import GesturePredictor
    from .gestures import get_gesture_info, draw_text, GESTURE_CATALOG
except ImportError:
    from config import DEFAULT_MODEL_PATH
    from landmarks import HandLandmarkExtractor
    from predictor import GesturePredictor
    from gestures import get_gesture_info, draw_text, GESTURE_CATALOG


def draw_overlay(frame: np.ndarray, stable_label: str, confidence: float,
                 show_help: bool, fps: float, is_flipped: bool) -> np.ndarray:
    """Draw a modern, readable HUD overlay with gesture information, flip mode, and full guide."""
    h, w = frame.shape[:2]

    # 1. Top HUD Banner
    banner_h = 95
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (w - 15, banner_h), (25, 25, 30), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    border_color = (60, 220, 90) if stable_label not in ("Unknown", "NoHand") else (0, 160, 255)
    if stable_label == "NoHand":
        border_color = (60, 60, 230)
    cv2.rectangle(frame, (15, 15), (w - 15, banner_h), border_color, 2)

    if stable_label == "NoHand":
        frame = draw_text(frame, "CHƯA PHÁT HIỆN BÀN TAY", (35, 30),
                          font_size=24, color=(80, 100, 255))
        frame = draw_text(frame, "Đưa bàn tay vào trước ống kính camera để nhận diện cử chỉ", (35, 62),
                          font_size=17, color=(200, 200, 200))
    elif stable_label == "Unknown":
        frame = draw_text(frame, f"Đang nhận diện cử chỉ... ({confidence:.1%})", (35, 30),
                          font_size=24, color=(0, 200, 255))
        frame = draw_text(frame, "Giữ tay ổn định trong khung hình hoặc nhấn H để xem danh sách cử chỉ", (35, 62),
                          font_size=17, color=(200, 200, 200))
    else:
        info = get_gesture_info(stable_label)
        main_title = f"{info.display_name.upper()}  -  {confidence:.1%}"
        frame = draw_text(frame, main_title, (35, 26),
                          font_size=26, color=(80, 255, 120))

        sub_title = f"Ý nghĩa: {info.vietnamese}"
        frame = draw_text(frame, sub_title, (35, 62),
                          font_size=17, color=(240, 240, 240))

        # Confidence bar at top-right
        bar_x1, bar_x2 = max(w - 220, 300), w - 35
        bar_y1, bar_y2 = 38, 54
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
        fill_w = int((bar_x2 - bar_x1) * min(max(confidence, 0.0), 1.0))
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (40, 220, 90), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (200, 200, 200), 1)

    # 2. Bottom status bar
    bottom_y = h - 35
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (15, bottom_y), (w - 15, h - 10), (20, 20, 25), -1)
    cv2.addWeighted(bottom_overlay, 0.75, frame, 0.25, 0, frame)

    flip_status = "BẬT" if is_flipped else "TẮT"
    controls_text = f"FPS: {fps:.0f}  |  Q/Esc: Thoát  |  H: {'Ẩn' if show_help else 'Xem'} hướng dẫn  |  M: Lật [{flip_status}]"
    frame = draw_text(frame, controls_text, (25, bottom_y + 4), font_size=16, color=(220, 220, 220))

    # 3. Comprehensive Help Guide Overlay (toggled with H)
    if show_help:
        guide_w = min(560, w - 40)
        guide_h = min(h - 160, 500)
        gx1 = max(20, w - guide_w - 20)
        gy1 = 110
        gx2 = gx1 + guide_w
        gy2 = gy1 + guide_h

        guide_overlay = frame.copy()
        cv2.rectangle(guide_overlay, (gx1, gy1), (gx2, gy2), (18, 20, 24), -1)
        cv2.addWeighted(guide_overlay, 0.90, frame, 0.10, 0, frame)
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (80, 180, 255), 2)

        frame = draw_text(frame, "DANH SÁCH TẤT CẢ CÁC CỬ CHỈ ĐƯỢC HỖ TRỢ", (gx1 + 15, gy1 + 10),
                          font_size=18, color=(255, 220, 60))

        # Compact 2-column list of all gestures
        col1 = [
            ("Like", "Ngón cái giơ lên (Thích)"),
            ("A", "Nắm đấm, cái thẳng cạnh trỏ"),
            ("B", "4 ngón thẳng, cái gập ngang"),
            ("C", "Khum tay hình chữ C"),
            ("D", "Ngón trỏ chỉ thẳng lên"),
            ("E", "Gập các đầu ngón móng vuốt"),
            ("F / OK", "Vòng tròn cái + trỏ, 3 ngón xòe"),
            ("G", "Cái & trỏ chỉ ngang song song"),
            ("H", "Trỏ & giữa duỗi ngang song song"),
            ("I", "Chỉ ngón út thẳng lên"),
            ("J", "Ngón út vẽ nét chữ J"),
            ("K", "Trỏ lên, giữa hướng trước"),
            ("L", "Góc vuông chữ L (90 độ)"),
            ("M", "Ba ngón gập đè ngón cái"),
            ("N", "Hai ngón gập đè ngón cái"),
        ]

        col2 = [
            ("O", "Khum tròn các ngón chữ O"),
            ("P", "Chữ K chúc đầu ngón xuống"),
            ("Q", "Cái & trỏ chúc xuống dưới"),
            ("R", "Bắt chéo trỏ & giữa (May mắn)"),
            ("S / Fist", "Nắm đấm kín (Nắm đấm)"),
            ("T", "Ngón cái kẹp giữa trỏ & giữa"),
            ("U", "Hai ngón chụm thẳng lên"),
            ("V / Peace", "Hai ngón chữ V (Hòa bình)"),
            ("W / Three", "Ba ngón xòe thẳng (Số 3)"),
            ("X / Hook", "Ngón trỏ co gập lưỡi câu"),
            ("Y / Call", "Ngón cái & út giơ ra (Alo)"),
            ("Z / Point", "Chỉ ngón trỏ vẽ chữ Z"),
            ("Space", "Bàn tay mở 5 ngón (Dừng lại)"),
            ("Dislike", "Ngón cái chúc xuống dưới"),
            ("Rock", "Ngón trỏ & út giơ thẳng"),
        ]

        # Draw Col 1
        y_offset = gy1 + 40
        col2_x = gx1 + int(guide_w * 0.50)
        for code, desc in col1:
            if y_offset + 22 > gy2 - 5:
                break
            frame = draw_text(frame, f"• {code}:", (gx1 + 12, y_offset), font_size=13, color=(100, 240, 160))
            frame = draw_text(frame, desc, (gx1 + 80, y_offset), font_size=13, color=(230, 230, 230))
            y_offset += 24

        # Draw Col 2
        y_offset = gy1 + 40
        for code, desc in col2:
            if y_offset + 22 > gy2 - 5:
                break
            frame = draw_text(frame, f"• {code}:", (col2_x, y_offset), font_size=13, color=(100, 240, 160))
            frame = draw_text(frame, desc, (col2_x + 75, y_offset), font_size=13, color=(230, 230, 230))
            y_offset += 24

    return frame


def run(model_path: Path | str, camera: int = 0, threshold: float = 0.60,
        smooth_window: int = 5, mirror: bool = True):
    predictor = GesturePredictor(model_path, confidence_threshold=threshold)
    extractor = HandLandmarkExtractor(static_image_mode=False, max_num_hands=1,
                                      min_detection_confidence=0.55, min_tracking_confidence=0.55)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Không thể mở webcam index {camera}. Hãy kiểm tra camera kết nối máy tính.")

    history = deque(maxlen=max(1, smooth_window))
    conf_history = deque(maxlen=max(1, smooth_window))
    show_help = False

    fps = 30.0
    prev_time = cv2.getTickCount()

    print("=" * 65)
    print("SignFlow - Hệ thống nhận diện cử chỉ tay thời gian thực")
    print(f"Tổng số lớp nhận diện: {len(predictor.class_names)} cử chỉ (A-Z, Like, Space, Del, v.v.)")
    print("Phím tắt: Q/Esc: Thoát | H: Xem hướng dẫn | M: Lật camera")
    print("=" * 65)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Apply flip (lật hình)
            if mirror:
                frame = cv2.flip(frame, 1)

            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            prev_time = curr_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff)

            features, result = extractor.extract(frame)
            extractor.draw(frame, result)

            if features is not None:
                label, confidence = predictor.predict(features)
                if label != 'Unknown':
                    history.append(label)
                    conf_history.append(confidence)
                else:
                    history.append(None)
                    conf_history.append(confidence)

                valid = [x for x in history if x is not None]
                if valid:
                    stable_label = Counter(valid).most_common(1)[0][0]
                    avg_conf = float(np.mean([c for l, c in zip(history, conf_history) if l == stable_label]))
                else:
                    stable_label = 'Unknown'
                    avg_conf = float(np.mean(conf_history)) if conf_history else 0.0

                frame = draw_overlay(frame, stable_label, avg_conf, show_help, fps, mirror)
            else:
                history.clear()
                conf_history.clear()
                frame = draw_overlay(frame, "NoHand", 0.0, show_help, fps, mirror)

            cv2.imshow('SignFlow - Hand Gesture Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or Esc to exit
                break
            elif key in (ord('h'), ord('H')):  # H to toggle help
                show_help = not show_help
            elif key in (ord('m'), ord('M')):  # M to toggle flip (lật)
                mirror = not mirror
                print(f"Chế độ lật camera: {'BẬT' if mirror else 'TẮT'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Đã tắt webcam và giải phóng tài nguyên.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chương trình nhận diện cử chỉ tay thời gian thực qua webcam.")
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH, help='Đường dẫn file mô hình .pth')
    parser.add_argument('--camera', type=int, default=0, help='Chỉ số camera (mặc định 0)')
    parser.add_argument('--threshold', type=float, default=0.60, help='Ngưỡng tin cậy tối thiểu (mặc định 0.60)')
    parser.add_argument('--smooth-window', type=int, default=5, help='Số khung hình làm mượt kết quả')
    parser.add_argument('--no-flip', '--no-mirror', dest='no_flip', action='store_true', help='Không lật hình camera')
    args = parser.parse_args()

    run(args.model, args.camera, args.threshold, args.smooth_window, not args.no_flip)
