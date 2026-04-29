"""Annotation primitives — OpenCV drawing for the annotated MP4 output.

Per docs/annotation/VIDEO_ANNOTATION_SPEC.md. Functions mutate the input
frame in place and return it for chaining (single-pass pipeline; perf > safety).
"""
from __future__ import annotations

from typing import Iterable, Literal

import cv2
import numpy as np

# --- Palette (locked) ---------------------------------------------------


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


PRIMARY = _bgr("#2E86AB")
ACCENT = _bgr("#F4A261")
LEFT_POSE = _bgr("#1f77b4")
RIGHT_POSE = _bgr("#ff7f0e")
ATHLETE_BBOX = _bgr("#FFFFFF")
BALL = _bgr("#E76F51")
CONE = _bgr("#FFB400")
GATE_ACTIVE = _bgr("#2A9D8F")
GATE_PASSED = _bgr("#264653")
GATE_PENDING = (128, 128, 128)
HUD_BG = (0, 0, 0)
HUD_TEXT = (255, 255, 255)

# --- COCO 17 skeleton edges ---------------------------------------------

LEFT_EDGES = [(5, 7), (7, 9), (11, 13), (13, 15)]            # left arm + left leg
RIGHT_EDGES = [(6, 8), (8, 10), (12, 14), (14, 16)]          # right arm + right leg
CENTER_EDGES = [
    (5, 6), (11, 12), (5, 11), (6, 12),                       # shoulders, hips, trunk
    (0, 1), (0, 2), (1, 3), (2, 4),                           # face
]

_LEFT_KP_INDICES = {1, 3, 5, 7, 9, 11, 13, 15}
_RIGHT_KP_INDICES = {2, 4, 6, 8, 10, 12, 14, 16}


# --- Drawing primitives -------------------------------------------------


def draw_skeleton(
    frame: np.ndarray, keypoints: np.ndarray, conf_threshold: float = 0.3
) -> np.ndarray:
    """Draw a COCO 17-point pose skeleton with left/right colour separation.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame, mutated in place.
    keypoints : np.ndarray, shape (17, 3)
        (x, y, confidence) per COCO keypoint.
    conf_threshold : float
        Skip keypoints / bones with confidence below this.
    """
    for u, v in LEFT_EDGES:
        _draw_bone(frame, keypoints, u, v, LEFT_POSE, conf_threshold)
    for u, v in RIGHT_EDGES:
        _draw_bone(frame, keypoints, u, v, RIGHT_POSE, conf_threshold)
    for u, v in CENTER_EDGES:
        _draw_bone(frame, keypoints, u, v, PRIMARY, conf_threshold)

    for i in range(17):
        if keypoints[i, 2] < conf_threshold:
            continue
        cx, cy = int(keypoints[i, 0]), int(keypoints[i, 1])
        color = (
            LEFT_POSE if i in _LEFT_KP_INDICES
            else RIGHT_POSE if i in _RIGHT_KP_INDICES
            else PRIMARY
        )
        cv2.circle(frame, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)
    return frame


def _draw_bone(
    frame: np.ndarray,
    kp: np.ndarray,
    u: int,
    v: int,
    color: tuple[int, int, int],
    threshold: float,
) -> None:
    if kp[u, 2] < threshold or kp[v, 2] < threshold:
        return
    p1 = (int(kp[u, 0]), int(kp[u, 1]))
    p2 = (int(kp[v, 0]), int(kp[v, 1]))
    cv2.line(frame, p1, p2, color, 2, lineType=cv2.LINE_AA)


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    label: str | None = None,
    color: tuple[int, int, int] = ATHLETE_BBOX,
) -> np.ndarray:
    """Draw a thin bounding box with optional label above."""
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(
            frame, label, (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return frame


def draw_gate(
    frame: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    state: Literal["active", "passed", "pending"] = "pending",
) -> np.ndarray:
    """Draw a gate / split line. State drives colour."""
    color = {"active": GATE_ACTIVE, "passed": GATE_PASSED, "pending": GATE_PENDING}[state]
    cv2.line(
        frame,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        color, 2, lineType=cv2.LINE_AA,
    )
    return frame


def draw_ball_trail(
    frame: np.ndarray,
    history: Iterable[tuple[float, float]],
    max_age: int = 30,
) -> np.ndarray:
    """Draw the recent ball positions, fading by age (newest brightest)."""
    pts = list(history)[-max_age:]
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        age = (n - 1 - i) / max(n - 1, 1)  # 0 for newest, 1 for oldest
        radius = max(2, int(6 * (1.0 - age)))
        cv2.circle(frame, (int(x), int(y)), radius, BALL, -1, lineType=cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    fields: dict[str, str],
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-left",
) -> np.ndarray:
    """Render a key-value HUD on a semi-transparent background."""
    if not fields:
        return frame
    pad = 12
    line_h = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 1

    rows = [(k, str(v)) for k, v in fields.items()]
    text_widths = [
        cv2.getTextSize(f"{k}: {v}", font, scale, thickness)[0][0] for k, v in rows
    ]
    box_w = max(text_widths) + pad * 2
    box_h = line_h * len(rows) + pad

    fh, fw = frame.shape[:2]
    x0, y0 = {
        "top-left": (12, 12),
        "top-right": (fw - 12 - box_w, 12),
        "bottom-left": (12, fh - 12 - box_h),
        "bottom-right": (fw - 12 - box_w, fh - 12 - box_h),
    }[position]

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

    for i, (label, value) in enumerate(rows):
        y = y0 + pad + (i + 1) * line_h - 8
        cv2.putText(
            frame, f"{label}: {value}",
            (x0 + pad, y), font, scale, HUD_TEXT, thickness, cv2.LINE_AA,
        )
    return frame


def event_flash(
    frame: np.ndarray,
    region_xyxy: np.ndarray,
    color: tuple[int, int, int] = ACCENT,
    intensity: float = 0.4,
) -> np.ndarray:
    """Tint a rectangular region by `intensity` (0..1) — for 3-frame event pulses."""
    x1, y1, x2, y2 = (int(v) for v in region_xyxy)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, intensity, frame, 1.0 - intensity, 0, dst=frame)
    return frame


def render_endcard(
    title: str,
    athlete: str,
    metric_rows: list[tuple[str, str, int]],
    test_score: int,
    band: str,
    size: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """Render a static end-card frame.

    Parameters
    ----------
    title : str
        Test display name.
    athlete : str
        Athlete identifier or profile name (no PII).
    metric_rows : list[tuple[str, str, int]]
        (display_label, value_string, score_0_100) per metric.
    test_score : int
        Aggregated test score (0..100).
    band : str
        Letter / band ('above_average', etc.).
    size : (W, H)
        Output frame size.

    Returns
    -------
    np.ndarray
        BGR frame, shape (H, W, 3).
    """
    w, h = size
    frame = np.full((h, w, 3), 12, dtype=np.uint8)

    cv2.putText(frame, title, (60, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, HUD_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Athlete: {athlete}", (60, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_TEXT, 1, cv2.LINE_AA)
    cv2.line(frame, (60, 155), (w - 60, 155), HUD_TEXT, 1, cv2.LINE_AA)

    y = 210
    for label, value_str, score in metric_rows:
        cv2.putText(frame, f"{label}: {value_str}", (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{score}/100", (w - 220, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, PRIMARY, 1, cv2.LINE_AA)
        y += 42

    cv2.line(frame, (60, y + 20), (w - 60, y + 20), HUD_TEXT, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Test score: {test_score}/100   Band: {band}",
        (60, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ACCENT, 2, cv2.LINE_AA,
    )
    return frame
