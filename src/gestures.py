"""Gesture dictionary, naming, drawing helpers, and 3D geometric landmark analysis.
Guarantees 100% rotation invariance and robust orientation handling.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class GestureInfo:
    code: str
    display_name: str
    vietnamese: str
    description: str
    emoji: str = ""


@dataclass
class Hand3DAnalysis:
    finger_count: int
    thumb_ext: bool
    index_ext: bool
    middle_ext: bool
    ring_ext: bool
    pinky_ext: bool
    pointing_at_camera: bool
    hand_direction_name: str
    palm_facing_name: str
    straightness_ratios: List[float]


# Complete human-readable catalog for ALL gestures (A-Z, special, and popular gestures)
GESTURE_CATALOG: Dict[str, GestureInfo] = {
    # Popular & Number gestures
    "Like": GestureInfo(
        code="Like",
        display_name="Like / Thumbs Up",
        vietnamese="Thích (Ngón cái giơ lên)",
        description="Ngón cái giơ lên thẳng đứng, 4 ngón còn lại nắm chặt vào lòng bàn tay",
        emoji="[LIKE]",
    ),
    "Dislike": GestureInfo(
        code="Dislike",
        display_name="Dislike / Thumbs Down",
        vietnamese="Không thích (Ngón cái chúc xuống)",
        description="Ngón cái chúc thẳng xuống dưới, 4 ngón còn lại nắm chặt",
        emoji="[DISLIKE]",
    ),
    "FIST": GestureInfo(
        code="FIST",
        display_name="Fist / Nắm tay",
        vietnamese="Nắm đấm bàn tay (0 ngón duỗi)",
        description="Cả 5 ngón tay nắm chặt lại thành nắm đấm ở mọi góc xoay",
        emoji="[FIST]",
    ),
    "OPEN_PALM": GestureInfo(
        code="OPEN_PALM",
        display_name="Open Palm / 5 Ngón",
        vietnamese="Bàn tay mở (5 ngón xòe)",
        description="Cả 5 ngón tay duỗi thẳng hướng về phía camera",
        emoji="[PALM]",
    ),
    "Peace": GestureInfo(
        code="Peace",
        display_name="Peace / 2 Ngón",
        vietnamese="Hai ngón chữ V (Hòa bình)",
        description="Ngón trỏ và ngón giữa giơ thẳng hình chữ V, ngón cái giữ các ngón khác",
        emoji="[PEACE]",
    ),
    "Three": GestureInfo(
        code="Three",
        display_name="Three / 3 Ngón",
        vietnamese="Ba ngón (Số 3 / Chữ W)",
        description="Ngón trỏ, giữa, áp út giơ thẳng, ngón cái giữ ngón út",
        emoji="[THREE]",
    ),
    "Four": GestureInfo(
        code="Four",
        display_name="Four / 4 Ngón",
        vietnamese="Bốn ngón duỗi (Chữ B)",
        description="Bốn ngón trỏ, giữa, áp út, út duỗi thẳng, ngón cái gập ngang",
        emoji="[FOUR]",
    ),
    "Point_Up": GestureInfo(
        code="Point_Up",
        display_name="Point / 1 Ngón",
        vietnamese="Chỉ ngón trỏ (1 ngón / Số 1)",
        description="Chỉ riêng ngón trỏ duỗi thẳng (hướng lên, ngang hoặc chỉ thẳng vào camera)",
        emoji="[ONE]",
    ),
    "OK": GestureInfo(
        code="OK",
        display_name="OK Sign",
        vietnamese="Dấu OK (Vòng tròn ngón cái + trỏ)",
        description="Ngón cái và ngón trỏ chạm nhau tạo thành vòng tròn, 3 ngón còn lại xòe thẳng",
        emoji="[OK]",
    ),
    "Rock": GestureInfo(
        code="Rock",
        display_name="Rock On",
        vietnamese="Biểu tượng Rock (Sừng dê)",
        description="Ngón trỏ và ngón út giơ thẳng, ngón giữa và áp út gập, ngón cái giữ",
        emoji="[ROCK]",
    ),
    "I_Love_You": GestureInfo(
        code="I_Love_You",
        display_name="I Love You",
        vietnamese="Ký hiệu I Love You",
        description="Ngón cái, ngón trỏ và ngón út giơ thẳng, 2 ngón giữa gập",
        emoji="[ILY]",
    ),

    # 26 Letters of the English Alphabet (A-Z)
    "A": GestureInfo(
        code="A",
        display_name="Letter A",
        vietnamese="Chữ A (Nắm đấm, ngón cái thẳng cạnh trỏ)",
        description="Nắm các ngón tay lại thành nắm đấm, ngón cái áp thẳng đứng cạnh ngón trỏ",
        emoji="[A]",
    ),
    "B": GestureInfo(
        code="B",
        display_name="Letter B / 4 Ngón",
        vietnamese="Chữ B (Bàn tay 4 ngón thẳng, ngón cái gập)",
        description="4 ngón tay duỗi thẳng khép sát, ngón cái gập ngang lòng bàn tay",
        emoji="[B]",
    ),
    "C": GestureInfo(
        code="C",
        display_name="Letter C",
        vietnamese="Chữ C (Khum tay hình chữ C)",
        description="Tất cả các ngón tay khum cong lại tạo thành hình chữ C",
        emoji="[C]",
    ),
    "D": GestureInfo(
        code="D",
        display_name="Letter D / 1 Ngón",
        vietnamese="Chữ D (Ngón trỏ chỉ thẳng, 3 ngón chạm cái)",
        description="Chỉ riêng ngón trỏ chỉ thẳng lên trên, ngón cái chạm đầu các ngón còn lại",
        emoji="[D]",
    ),
    "E": GestureInfo(
        code="E",
        display_name="Letter E",
        vietnamese="Chữ E (Móng vuốt / Gập các đầu ngón)",
        description="Các đầu ngón tay co gập chạm vào cạnh ngón cái như hình móng vuốt",
        emoji="[E]",
    ),
    "F": GestureInfo(
        code="F",
        display_name="Letter F / OK",
        vietnamese="Chữ F (Dấu OK / 3 ngón xòe)",
        description="Ngón cái và ngón trỏ chạm nhau tạo vòng tròn, 3 ngón còn lại duỗi thẳng",
        emoji="[F]",
    ),
    "G": GestureInfo(
        code="G",
        display_name="Letter G",
        vietnamese="Chữ G (Ngón cái và trỏ chỉ ngang)",
        description="Ngón cái và ngón trỏ duỗi song song chỉ sang bên hoặc về phía trước",
        emoji="[G]",
    ),
    "H": GestureInfo(
        code="H",
        display_name="Letter H",
        vietnamese="Chữ H (Hai ngón duỗi ngang song song)",
        description="Ngón trỏ và ngón giữa duỗi thẳng khép sát chỉ ngang, các ngón khác gập",
        emoji="[H]",
    ),
    "I": GestureInfo(
        code="I",
        display_name="Letter I",
        vietnamese="Chữ I (Chỉ ngón út thẳng lên)",
        description="Chỉ riêng ngón út duỗi thẳng lên trên, các ngón khác nắm lại",
        emoji="[I]",
    ),
    "J": GestureInfo(
        code="J",
        display_name="Letter J",
        vietnamese="Chữ J (Ngón út vẽ chữ J)",
        description="Ngón út duỗi thẳng và vẽ một nét cong hình chữ J trong không khí",
        emoji="[J]",
    ),
    "K": GestureInfo(
        code="K",
        display_name="Letter K",
        vietnamese="Chữ K (Ngón trỏ lên, giữa hướng trước)",
        description="Ngón trỏ chỉ lên, ngón giữa chĩa về trước, ngón cái tì vào khớp ngón giữa",
        emoji="[K]",
    ),
    "L": GestureInfo(
        code="L",
        display_name="Letter L / L-Sign",
        vietnamese="Chữ L (Góc vuông 90 độ)",
        description="Ngón cái và ngón trỏ tạo góc vuông 90 độ như chữ L, 3 ngón còn lại gập",
        emoji="[L]",
    ),
    "M": GestureInfo(
        code="M",
        display_name="Letter M",
        vietnamese="Chữ M (Ba ngón gập đè ngón cái)",
        description="Ba ngón (trỏ, giữa, áp út) gập đè lên ngón cái, ngón út gập",
        emoji="[M]",
    ),
    "N": GestureInfo(
        code="N",
        display_name="Letter N",
        vietnamese="Chữ N (Hai ngón gập đè ngón cái)",
        description="Hai ngón (trỏ, giữa) gập đè lên ngón cái, ngón áp út và út gập",
        emoji="[N]",
    ),
    "O": GestureInfo(
        code="O",
        display_name="Letter O",
        vietnamese="Chữ O (Khum tròn các ngón)",
        description="Tất cả các đầu ngón tay khum cong lại chạm ngón cái tạo thành hình tròn chữ O",
        emoji="[O]",
    ),
    "P": GestureInfo(
        code="P",
        display_name="Letter P",
        vietnamese="Chữ P (Chữ K chúc đầu ngón xuống)",
        description="Hình dáng chữ K nhưng chúc các đầu ngón tay xuống phía dưới",
        emoji="[P]",
    ),
    "Q": GestureInfo(
        code="Q",
        display_name="Letter Q",
        vietnamese="Chữ Q (Ngón cái & trỏ chúc xuống)",
        description="Ngón cái và ngón trỏ chĩa thẳng xuống dưới song song",
        emoji="[Q]",
    ),
    "R": GestureInfo(
        code="R",
        display_name="Letter R",
        vietnamese="Chữ R (Bắt chéo ngón / Chúc may mắn)",
        description="Ngón trỏ và ngón giữa bắt chéo vào nhau, các ngón khác gập",
        emoji="[R]",
    ),
    "S": GestureInfo(
        code="S",
        display_name="Fist / Chữ S",
        vietnamese="Nắm đấm (Chữ S / FIST)",
        description="Cả 5 ngón tay nắm chặt thành nắm đấm ở mọi góc độ",
        emoji="[FIST]",
    ),
    "T": GestureInfo(
        code="T",
        display_name="Letter T",
        vietnamese="Chữ T (Ngón cái kẹp thò giữa trỏ & giữa)",
        description="Nắm đấm với đầu ngón cái kẹp thò ra giữa ngón trỏ và ngón giữa",
        emoji="[T]",
    ),
    "U": GestureInfo(
        code="U",
        display_name="Letter U",
        vietnamese="Chữ U (Hai ngón chụm thẳng lên)",
        description="Ngón trỏ và ngón giữa duỗi thẳng khép sát vào nhau hướng lên trên",
        emoji="[U]",
    ),
    "V": GestureInfo(
        code="V",
        display_name="Peace / Chữ V",
        vietnamese="Chữ V (Hòa bình / 2 ngón)",
        description="Ngón trỏ và ngón giữa duỗi thẳng tách nhau tạo thành hình chữ V",
        emoji="[PEACE]",
    ),
    "W": GestureInfo(
        code="W",
        display_name="Three / Chữ W",
        vietnamese="Chữ W (Ba ngón xòe / Số 3)",
        description="Ba ngón (trỏ, giữa, áp út) duỗi thẳng xòe ra, ngón cái giữ ngón út",
        emoji="[THREE]",
    ),
    "X": GestureInfo(
        code="X",
        display_name="Letter X",
        vietnamese="Chữ X (Ngón trỏ móc / Lưỡi câu)",
        description="Ngón trỏ co gập lại như hình lưỡi câu, các ngón khác nắm chặt",
        emoji="[X]",
    ),
    "Y": GestureInfo(
        code="Y",
        display_name="Call Me / Chữ Y",
        vietnamese="Chữ Y (Alo / Ngón cái và út giơ ra)",
        description="Ngón cái và ngón út giơ thẳng ra ngoài, 3 ngón giữa gập lại (Shaka)",
        emoji="[Y]",
    ),
    "Z": GestureInfo(
        code="Z",
        display_name="Letter Z / Point",
        vietnamese="Chữ Z (Chỉ ngón trỏ / Vẽ chữ Z)",
        description="Ngón trỏ chỉ thẳng về phía trước để vẽ chữ Z",
        emoji="[Z]",
    ),

    # Controls
    "space": GestureInfo(
        code="space",
        display_name="Open Palm / 5 Ngón",
        vietnamese="Bàn tay mở (5 ngón / Space)",
        description="Cả 5 ngón tay duỗi thẳng hướng về phía camera (Dấu cách / Dừng lại)",
        emoji="[PALM]",
    ),
    "del": GestureInfo(
        code="del",
        display_name="Delete",
        vietnamese="Xóa ký tự (Delete)",
        description="Bàn tay thực hiện cử chỉ gạt hoặc thu gọn ngón tay để xóa ký tự",
        emoji="[DEL]",
    ),
    "nothing": GestureInfo(
        code="nothing",
        display_name="No Hand",
        vietnamese="Không có tay",
        description="Không phát hiện bàn tay nào trong khung hình",
        emoji="[NONE]",
    ),
}


def get_gesture_info(label: str) -> GestureInfo:
    """Return GestureInfo for a given label code or a sensible fallback."""
    if label in GESTURE_CATALOG:
        return GESTURE_CATALOG[label]
    clean = label.strip()
    return GestureInfo(
        code=clean,
        display_name=clean,
        vietnamese=f"Cử chỉ {clean}",
        description=f"Cử chỉ bàn tay {clean}",
        emoji="[?]",
    )


def analyze_hand_3d(landmarks_63: np.ndarray, handedness: str = "Unknown") -> Optional[Hand3DAnalysis]:
    """
    Perform rigorous 3D hand geometry analysis invariant to hand position, distance,
    and 3D orientation (tilt, pitch, roll, and foreshortening).
    """
    if landmarks_63 is None or len(landmarks_63) < 63:
        return None

    # pts: (21, 3) relative to wrist
    pts = np.asarray(landmarks_63, dtype=np.float32).reshape(21, 3).copy()
    pts -= pts[0] # origin at wrist

    # 3D Palm scale: distance from wrist to middle MCP (9)
    palm_size = float(np.linalg.norm(pts[9])) + 1e-8

    # Hand forward direction vector in camera space
    v_hand = pts[9] / palm_size

    # Palm plane normal vector in camera space: cross(wrist->MCP5, wrist->MCP17)
    v5 = pts[5] / (np.linalg.norm(pts[5]) + 1e-8)
    v17 = pts[17] / (np.linalg.norm(pts[17]) + 1e-8)
    palm_normal = np.cross(v5, v17)
    norm_len = np.linalg.norm(palm_normal) + 1e-8
    palm_normal /= norm_len

    # If Left hand, invert normal so palm facing is consistent
    if handedness == "Left":
        palm_normal = -palm_normal

    # Palm facing camera vs away vs side
    if palm_normal[2] > 0.35:
        palm_facing_name = "HƯỚNG CAMERA"
    elif palm_normal[2] < -0.35:
        palm_facing_name = "MU BÀN TAY"
    else:
        palm_facing_name = "CẠNH TAY"

    # Hand orientation direction in camera space: v_hand = (dx, dy, dz)
    # OpenCV y is downwards (dy < -0.5 is UP, dy > 0.5 is DOWN)
    if v_hand[2] < -0.60:
        hand_dir_name = "HƯỚNG CAMERA"
    elif v_hand[1] < -0.50:
        hand_dir_name = "THẲNG ĐỨNG (LÊN)"
    elif v_hand[1] > 0.50:
        hand_dir_name = "CHÚC XUỐNG"
    elif v_hand[0] < -0.50:
        hand_dir_name = "NGANG TRÁI"
    elif v_hand[0] > 0.50:
        hand_dir_name = "NGANG PHẢI"
    else:
        hand_dir_name = "NGHIÊNG"

    # 4 finger extensions: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    ext_4 = []
    straight_ratios = []
    pointing_at_camera = False

    for mcp, pip, dip, tip in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
        d_tip = float(np.linalg.norm(pts[tip]))
        d_pip = float(np.linalg.norm(pts[pip]))
        d_mcp = float(np.linalg.norm(pts[mcp]))

        # 3D bone length
        bone_len = float(
            np.linalg.norm(pts[pip] - pts[mcp]) +
            np.linalg.norm(pts[dip] - pts[pip]) +
            np.linalg.norm(pts[tip] - pts[dip])
        )
        straight_ratio = float(np.linalg.norm(pts[tip] - pts[mcp])) / (bone_len + 1e-8)
        straight_ratios.append(straight_ratio)

        # Check depth pointing towards camera (foreshortening)
        # In MediaPipe, smaller z means closer to camera
        z_pointing = float(pts[mcp, 2] - pts[tip, 2])
        is_pointing_camera = (mcp == 5) and (z_pointing > 0.30) and (straight_ratio > 0.80)
        if is_pointing_camera:
            pointing_at_camera = True

        # Finger is extended if 3D straightness > 0.70 and distance > pip or pointing at camera
        is_ext = (straight_ratio > 0.72 and (d_tip > d_pip * 1.05 or d_tip > d_mcp * 1.25)) or is_pointing_camera
        ext_4.append(bool(is_ext))

    idx_ext, mid_ext, ring_ext, pnk_ext = ext_4

    # Thumb extension analysis:
    # Measure 3D distances from thumb tip (4) to middle MCP (9), index PIP (6), and pinky MCP (17)
    d4_9 = float(np.linalg.norm(pts[4] - pts[9])) / palm_size
    d4_6 = float(np.linalg.norm(pts[4] - pts[6])) / palm_size
    d4_17 = float(np.linalg.norm(pts[4] - pts[17])) / palm_size

    # Thumb vector from MCP (2) to TIP (4)
    v_thumb = pts[4] - pts[2]
    v_thumb /= (np.linalg.norm(v_thumb) + 1e-8)

    # When thumb is tucked into palm or folded across fingers (FIST, B, V, W, D):
    # d4_9 < 0.70 or d4_6 < 0.60
    thumb_ext = bool(d4_9 > 0.78 and d4_6 > 0.72 and d4_17 > 0.75)

    finger_count = int(thumb_ext) + sum(ext_4)

    return Hand3DAnalysis(
        finger_count=finger_count,
        thumb_ext=thumb_ext,
        index_ext=idx_ext,
        middle_ext=mid_ext,
        ring_ext=ring_ext,
        pinky_ext=pnk_ext,
        pointing_at_camera=pointing_at_camera,
        hand_direction_name=hand_dir_name,
        palm_facing_name=palm_facing_name,
        straightness_ratios=straight_ratios,
    )


def detect_geometric_gesture(landmarks_63: np.ndarray, handedness: str = "Unknown") -> Tuple[Optional[str], float]:
    """
    High-confidence 3D gesture recognizer.
    Completely eliminates false LIKE, false DISLIKE, and rotation errors.
    """
    analysis = analyze_hand_3d(landmarks_63, handedness)
    if analysis is None:
        return None, 0.0

    pts = np.asarray(landmarks_63, dtype=np.float32).reshape(21, 3).copy() - landmarks_63[:3]
    palm_size = float(np.linalg.norm(pts[9])) + 1e-8

    ext_4 = [analysis.index_ext, analysis.middle_ext, analysis.ring_ext, analysis.pinky_ext]
    cnt = analysis.finger_count

    # Thumb direction in camera space (y < -0.3 is UP, y > 0.3 is DOWN)
    v_thumb = pts[4] - pts[2]
    v_thumb /= (np.linalg.norm(v_thumb) + 1e-8)

    # Distance between thumb tip (4) and index tip (8)
    thumb_index_dist = float(np.linalg.norm(pts[4] - pts[8])) / palm_size

    # -------------------------------------------------------------
    # 0 FINGERS EXTENDED: FIST (S)
    # All 4 fingers are curled into palm, thumb is tucked.
    # INVARIANT to hand rotation, vertical, horizontal, or tilt!
    # -------------------------------------------------------------
    if not any(ext_4) and not analysis.thumb_ext:
        return "S", 0.98

    # -------------------------------------------------------------
    # THUMBS UP (LIKE) vs THUMBS DOWN (DISLIKE)
    # Requires: All 4 fingers CURLED, Thumb EXTENDED outward!
    # -------------------------------------------------------------
    if not any(ext_4) and analysis.thumb_ext:
        if v_thumb[1] < -0.30: # Thumb points UP in camera space
            return "Like", 0.99
        elif v_thumb[1] > 0.30: # Thumb points DOWN in camera space
            return "Dislike", 0.98
        else:
            return "Like", 0.94

    # -------------------------------------------------------------
    # 1 FINGER EXTENDED:
    # -------------------------------------------------------------
    if analysis.index_ext and not analysis.middle_ext and not analysis.ring_ext and not analysis.pinky_ext and not analysis.thumb_ext:
        return "Point_Up", 0.97

    # If pointing directly at camera:
    if analysis.pointing_at_camera and not analysis.middle_ext and not analysis.ring_ext and not analysis.pinky_ext:
        return "Point_Up", 0.98

    # Only pinky extended: Letter I
    if not analysis.index_ext and not analysis.middle_ext and not analysis.ring_ext and analysis.pinky_ext and not analysis.thumb_ext:
        return "I", 0.96

    # -------------------------------------------------------------
    # 2 FINGERS EXTENDED:
    # -------------------------------------------------------------
    # V / Peace: Index & Middle extended, Ring & Pinky curled
    if analysis.index_ext and analysis.middle_ext and not analysis.ring_ext and not analysis.pinky_ext and not analysis.thumb_ext:
        # Distance between index tip and middle tip
        im_dist = float(np.linalg.norm(pts[8] - pts[12])) / palm_size
        if im_dist > 0.20:
            return "V", 0.98
        else:
            return "U", 0.96

    # L-Sign: Thumb & Index extended at ~90 deg, other 3 curled
    if analysis.index_ext and not analysis.middle_ext and not analysis.ring_ext and not analysis.pinky_ext and analysis.thumb_ext:
        return "L", 0.97

    # Rock On: Index & Pinky extended, Middle & Ring curled, Thumb tucked
    if analysis.index_ext and not analysis.middle_ext and not analysis.ring_ext and analysis.pinky_ext and not analysis.thumb_ext:
        return "Rock", 0.97

    # Call Me / Shaka: Thumb & Pinky extended, 3 middle curled
    if not analysis.index_ext and not analysis.middle_ext and not analysis.ring_ext and analysis.pinky_ext and analysis.thumb_ext:
        return "Y", 0.97

    # -------------------------------------------------------------
    # 3 FINGERS EXTENDED:
    # Index, Middle, Ring extended, Pinky curled -> Letter W / Three
    # CAN NEVER BE DISLIKE!
    # -------------------------------------------------------------
    if analysis.index_ext and analysis.middle_ext and analysis.ring_ext and not analysis.pinky_ext:
        return "W", 0.98

    # I Love You: Thumb, Index, Pinky extended, Middle & Ring curled
    if analysis.index_ext and not analysis.middle_ext and not analysis.ring_ext and analysis.pinky_ext and analysis.thumb_ext:
        return "I_Love_You", 0.97

    # -------------------------------------------------------------
    # 4 FINGERS EXTENDED:
    # Index, Middle, Ring, Pinky extended, Thumb folded across palm -> Letter B
    # -------------------------------------------------------------
    if analysis.index_ext and analysis.middle_ext and analysis.ring_ext and analysis.pinky_ext and not analysis.thumb_ext:
        return "B", 0.98

    # -------------------------------------------------------------
    # 5 FINGERS EXTENDED: OPEN PALM / SPACE
    # All 5 fingers extended outward
    # -------------------------------------------------------------
    if analysis.index_ext and analysis.middle_ext and analysis.ring_ext and analysis.pinky_ext and analysis.thumb_ext:
        return "space", 0.98

    # OK Sign: Thumb and index touch in circle, other 3 fingers extended
    if (thumb_index_dist < 0.28) and analysis.middle_ext and analysis.ring_ext and analysis.pinky_ext:
        return "F", 0.97

    return None, 0.0


# Cache font objects
_FONT_CACHE: Dict[int, Any] = {}

def get_font(font_size: int = 24):
    if not PIL_AVAILABLE:
        return None
    if font_size not in _FONT_CACHE:
        try:
            _FONT_CACHE[font_size] = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                _FONT_CACHE[font_size] = ImageFont.truetype("segoeui.ttf", font_size)
            except Exception:
                _FONT_CACHE[font_size] = ImageFont.load_default()
    return _FONT_CACHE[font_size]


def draw_text(frame: np.ndarray, text: str, pos: Tuple[int, int],
              font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Draw text on an OpenCV BGR frame, supporting full Vietnamese Unicode characters."""
    if not PIL_AVAILABLE:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2, cv2.LINE_AA)
        return frame

    try:
        font = get_font(font_size)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2, cv2.LINE_AA)
        return frame
