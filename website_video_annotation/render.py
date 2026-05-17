"""Render an annotated showcase video for the website.

Standalone one-off renderer. Not part of the test pipeline architecture
(no AthleteProfile, no normalization, no end-card). The goal here is a
visually polished broadcast-style overlay for marketing.

Run:
    uv run website_video_annotation/render.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.core.models.registry import get_model

ROOT = Path(__file__).resolve().parent

# Default mapping of video stem → source filename. The output is always
# written to `<stem>_annotated.mp4` next to the source.
VIDEOS: dict[str, str] = {
    "match": "6077718-uhd_3840_2160_25fps.mp4",
    "player_dribbling": "player_dribbling.mp4",
}

# Render resolution. Match source (3840x2160 / 4K).
OUT_W, OUT_H = 3840, 2160
S = OUT_H / 1080  # uniform scale factor; all visual sizes derive from this


def px(v: float) -> int:
    return int(round(v * S))

# Palette (BGR for OpenCV, RGBA for PIL).
TEAL = (0xF1, 0xC4, 0x2E)            # BGR for #2EC4F1
TEAL_RGB = (0x2E, 0xC4, 0xF1)
ORANGE = (0x61, 0xA2, 0xF4)          # BGR for #F4A261
ORANGE_RGB = (0xF4, 0xA2, 0x61)
BALL_RGB = (0xE7, 0x6F, 0x51)
WHITE = (255, 255, 255)
SHADOW = (0, 0, 0)
DOME = (210, 210, 210)

FONT_DIR = "/System/Library/Fonts"
FONT_REGULAR = f"{FONT_DIR}/HelveticaNeue.ttc"
FONT_BOLD = f"{FONT_DIR}/HelveticaNeue.ttc"


# --- Data containers ----------------------------------------------------


@dataclass
class FrameData:
    persons: list[dict]  # [{tid, bbox_xyxy, keypoints (17,3)}]
    ball: tuple[float, float] | None


# --- Detection pass -----------------------------------------------------


def detect_pass(src: Path, scale: tuple[int, int]) -> tuple[list[FrameData], float, int]:
    """Single pass over the video: pose tracking + ball detection."""
    pose = get_model("pose_default")
    detector = get_model("object_detector")

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[FrameData] = []
    sw, sh = scale

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        # Resize to render resolution; do detection at this scale too.
        frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        # Pose with tracking (persist across frames).
        pres = pose.track(
            frame, persist=True, verbose=False, conf=0.30, iou=0.5,
            tracker="bytetrack.yaml", classes=[0],
        )[0]

        persons: list[dict] = []
        if pres.boxes is not None and len(pres.boxes) > 0:
            boxes = pres.boxes.xyxy.cpu().numpy()
            ids = (
                pres.boxes.id.cpu().numpy().astype(int)
                if pres.boxes.id is not None
                else np.arange(len(boxes))
            )
            kps = (
                pres.keypoints.data.cpu().numpy()
                if pres.keypoints is not None
                else np.zeros((len(boxes), 17, 3))
            )
            for b, tid, kp in zip(boxes, ids, kps):
                # "Whiteness" of the torso strip. The source clip is
                # underexposed against a bright dome — the white #10 jersey
                # reads as medium-gray, not bright white. So we cannot rely
                # on a high-V threshold; instead, we use desaturation as the
                # discriminator (white & gray are low-S; pink/blue/red are
                # high-S). Mean saturation inverted into a 0..1 score.
                x1, y1, x2, y2 = b.astype(int)
                ty1 = int(y1 + 0.30 * (y2 - y1))
                ty2 = int(y1 + 0.50 * (y2 - y1))
                tx1 = int(x1 + 0.20 * (x2 - x1))
                tx2 = int(x2 - 0.20 * (x2 - x1))
                white = 0.0
                if ty2 > ty1 and tx2 > tx1:
                    strip = frame[ty1:ty2, tx1:tx2]
                    if strip.size:
                        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
                        s = hsv[..., 1]
                        white = float(max(0.0, 1.0 - s.mean() / 80.0))
                persons.append({"tid": int(tid), "bbox": b, "kp": kp, "white": white})

        # Ball detection (COCO class 32 = sports_ball).
        dres = detector(frame, classes=[32], verbose=False, conf=0.25, iou=0.45)[0]
        ball = None
        if dres.boxes is not None and len(dres.boxes) > 0:
            bb = dres.boxes.xyxy.cpu().numpy()
            confs = dres.boxes.conf.cpu().numpy()
            best = int(confs.argmax())
            x1, y1, x2, y2 = bb[best]
            ball = (float((x1 + x2) / 2), float((y1 + y2) / 2))

        frames.append(FrameData(persons=persons, ball=ball))
        if i % 25 == 0:
            print(f"  detect {i}/{n}")

    cap.release()
    return frames, fps, n


# --- Hero player selection ---------------------------------------------


def hero_series(frames: list[FrameData]) -> list[dict | None]:
    """Per-frame hero pick — ID-agnostic.

    Score each candidate by bbox area, ball proximity, and continuity with
    the previous hero's bbox center. Robust to ByteTrack re-IDs when the
    camera pans, since we never rely on track-ID persistence.
    """
    out: list[dict | None] = []
    last_center: tuple[float, float] | None = None
    last_area: float | None = None
    last_white: float = 0.0

    for fd in frames:
        if not fd.persons:
            out.append(None)
            continue

        # White-jersey gate. Whiteness is now a 0..1 desaturation score.
        # 0.6+ = clearly white kit; 0.3 = ambiguous; <0.2 = coloured.
        # Once the hero is locked on white, we strictly prefer white
        # candidates and emit None when no clean candidate is available.
        max_white = max((p.get("white", 0.0) for p in fd.persons), default=0.0)
        if last_white > 0.45:
            white_cands = [p for p in fd.persons if p.get("white", 0.0) > 0.40]
            if white_cands:
                candidates = white_cands
            else:
                out.append(None)
                continue
        elif max_white > 0.55:
            candidates = [p for p in fd.persons if p.get("white", 0.0) > 0.40]
        else:
            candidates = fd.persons

        best, best_score = None, -1.0
        for p in candidates:
            x1, y1, x2, y2 = p["bbox"]
            area = float((x2 - x1) * (y2 - y1))
            if area < 4000:  # ignore far-away players
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            ball_bonus = 0.0
            if fd.ball:
                bx, by = fd.ball
                d = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
                ball_bonus = max(0.0, 1.0 - d / 700.0)

            continuity = 0.0
            if last_center is not None:
                dx = cx - last_center[0]
                dy = cy - last_center[1]
                d = (dx * dx + dy * dy) ** 0.5
                continuity = max(0.0, 1.0 - d / 350.0)
                if last_area:
                    ratio = min(area, last_area) / max(area, last_area)
                    continuity *= ratio

            jersey_bonus = p.get("white", 0.0)  # 0..1 white-pixel ratio
            # sqrt-area so a giant foreground bbox doesn't dominate over a
            # smaller hero with the right jersey + ball + continuity.
            score = (area ** 0.5) * (
                0.3 + 0.8 * ball_bonus + 1.4 * continuity + 1.8 * jersey_bonus
            )
            if score > best_score:
                best_score = score
                best = p

        if best is None:
            out.append(None)
            continue
        x1, y1, x2, y2 = best["bbox"]
        last_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        last_area = float((x2 - x1) * (y2 - y1))
        last_white = float(best.get("white", 0.0))
        out.append(best)
    return out


def fill_gaps(series: list[dict | None], max_gap: int = 12) -> list[dict | None]:
    """Carry-forward small gaps in the hero series."""
    last = None
    last_age = 0
    out: list[dict | None] = []
    for s in series:
        if s is not None:
            out.append(s)
            last = s
            last_age = 0
        elif last is not None and last_age < max_gap:
            out.append(last)
            last_age += 1
        else:
            out.append(None)
    return out


# --- Juggle counting ---------------------------------------------------


def count_juggles(frames: list["FrameData"], fps: float) -> tuple[list[int], list[float]]:
    """Count juggle touches from the ball's vertical trajectory.

    A touch happens when the ball's vertical velocity flips from falling
    (y increasing) to rising (y decreasing) — i.e., the ball reached a
    local low point and was kicked back up.

    Returns
    -------
    counts : list[int]
        Cumulative juggle count at each frame.
    tempo  : list[float]
        Touches per second over a 2.5 s sliding window.
    """
    n = len(frames)
    # Forward-fill missing ball detections with the last known y.
    ys: list[float | None] = []
    last = None
    miss = 0
    for fd in frames:
        if fd.ball is not None:
            last = fd.ball[1]
            miss = 0
            ys.append(last)
        elif last is not None and miss < int(fps * 0.5):
            miss += 1
            ys.append(last)
        else:
            ys.append(None)

    # Smooth where we have data (rolling median over ~6 frames).
    win = max(3, int(fps / 10))
    smooth: list[float | None] = []
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        vals = [v for v in ys[lo:hi] if v is not None]
        smooth.append(float(np.median(vals)) if vals else None)

    counts = [0] * n
    touch_frames: list[int] = []
    min_gap = max(4, int(fps * 0.20))   # ignore <0.20 s between touches
    min_dip = 8.0                       # px — minimum dip prominence
    running = 0
    for i in range(2, n):
        a, b, c = smooth[i - 2], smooth[i - 1], smooth[i]
        if a is None or b is None or c is None:
            counts[i] = running
            continue
        # Local maximum in y (= lowest point of the ball trajectory in
        # screen coords). Velocity flips: was non-negative, now negative.
        v_prev = b - a
        v_now = c - b
        if v_prev >= 0 and v_now < 0 and (b - min(a, c)) > -min_dip:
            if not touch_frames or (i - touch_frames[-1]) >= min_gap:
                touch_frames.append(i)
                running += 1
        counts[i] = running
    counts[0] = counts[2] if n > 2 else 0
    counts[1] = counts[2] if n > 2 else 0

    # Tempo: touches per second over a 2.5 s window.
    tempo: list[float] = []
    window_s = 2.5
    window_f = int(window_s * fps)
    for i in range(n):
        lo = max(0, i - window_f)
        recent = [t for t in touch_frames if lo <= t <= i]
        seconds = max(0.3, (i - lo) / fps)
        tempo.append(len(recent) / seconds)
    return counts, tempo


# --- Speed computation --------------------------------------------------


def smooth_speed(series: list[dict | None], fps: float) -> tuple[list[float], list[float]]:
    """Estimate horizontal speed in km/h using bbox height as scale.

    Returns (speed_kmh, peak_kmh_so_far).
    """
    centers: list[tuple[float, float] | None] = []
    heights: list[float | None] = []
    for s in series:
        if s is None:
            centers.append(None)
            heights.append(None)
            continue
        x1, y1, x2, y2 = s["bbox"]
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        heights.append(y2 - y1)

    # Rolling median height to stabilize scale.
    window = 9
    smooth_h: list[float] = []
    for i in range(len(heights)):
        lo, hi = max(0, i - window), min(len(heights), i + window + 1)
        vals = [h for h in heights[lo:hi] if h]
        smooth_h.append(float(np.median(vals)) if vals else 200.0)

    # Pixel velocity → m/s via (1.75 m / bbox_height_px).
    speeds_mps: list[float] = []
    for i in range(len(centers)):
        if i == 0 or centers[i] is None or centers[i - 1] is None:
            speeds_mps.append(0.0)
            continue
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        px = (dx * dx + dy * dy) ** 0.5
        m_per_px = 1.75 / max(smooth_h[i], 60.0)
        speeds_mps.append(px * m_per_px * fps)

    # Reject single-frame spikes (bbox jitter): cap any one frame to 1.6x of
    # the previous EMA value before smoothing.
    alpha = 0.25
    smooth_kmh: list[float] = []
    s = 0.0
    for v in speeds_mps:
        kmh = v * 3.6
        if s > 0.5:
            kmh = min(kmh, s * 1.6 + 1.0)
        s = alpha * kmh + (1 - alpha) * s
        smooth_kmh.append(s)

    # Clamp ceiling at realistic indoor-sprint upper bound.
    smooth_kmh = [min(v, 28.0) for v in smooth_kmh]

    peak: list[float] = []
    cur = 0.0
    for v in smooth_kmh:
        cur = max(cur, v)
        peak.append(cur)
    return smooth_kmh, peak


# --- Pose smoothing -----------------------------------------------------


def smooth_keypoints(series: list[dict | None]) -> list[np.ndarray | None]:
    """One-euro-ish: simple temporal smoothing per joint."""
    out: list[np.ndarray | None] = []
    prev: np.ndarray | None = None
    alpha = 0.55  # higher = more responsive
    for s in series:
        if s is None:
            out.append(None)
            prev = None
            continue
        kp = s["kp"].copy()
        if prev is not None and prev.shape == kp.shape:
            mask = (kp[:, 2] > 0.2) & (prev[:, 2] > 0.2)
            kp[mask, :2] = alpha * kp[mask, :2] + (1 - alpha) * prev[mask, :2]
        out.append(kp)
        prev = kp
    return out


# --- Glass-card rendering primitives (PIL) ------------------------------


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _glass_panel(
    base: Image.Image, xy: tuple[int, int, int, int], radius: int = 18, tint=(8, 14, 22, 175)
) -> None:
    """Frosted-glass panel: blur the background slice, darken it, round corners."""
    x1, y1, x2, y2 = xy
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(base.size[0], x2); y2 = min(base.size[1], y2)
    if x2 <= x1 or y2 <= y1:
        return

    # Blurred backdrop.
    crop = base.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=px(16)))
    # Darken / tint.
    tint_layer = Image.new("RGBA", crop.size, tint)
    crop = Image.alpha_composite(crop.convert("RGBA"), tint_layer)

    # Rounded mask.
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, crop.size[0], crop.size[1]), radius=radius, fill=255)

    base.paste(crop, (x1, y1), mask)

    # Hairline border.
    border = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        (x1, y1, x2 - 1, y2 - 1), radius=radius, outline=(255, 255, 255, 55), width=max(1, px(1))
    )
    base.alpha_composite(border)


def draw_juggle_readout(base: Image.Image, count: int, tempo: float) -> None:
    """Borderless juggle-count display, top-right.

    Big numeral (touches) + small "touches" tag + faint TEMPO subline +
    hairline meter showing tempo relative to a 2 touches/s reference.
    """
    W = base.size[0]
    pad = px(56)
    draw = ImageDraw.Draw(base)

    val = str(int(count))
    big = _font(px(120), bold=False)
    unit = _font(px(22), bold=True)
    sub_font = _font(px(16), bold=True)

    nbbox = draw.textbbox((0, 0), val, font=big)
    nw, nh = nbbox[2] - nbbox[0], nbbox[3] - nbbox[1]
    ubbox = draw.textbbox((0, 0), "touches", font=unit)
    uw = ubbox[2] - ubbox[0]
    tag_gap = px(14)

    right_edge = W - pad
    unit_x = right_edge - uw
    num_x = unit_x - tag_gap - nw
    y = pad

    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.text((num_x + px(3), y + px(3)), val, font=big, fill=(0, 0, 0, 180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(px(7)))
    base.alpha_composite(shadow)

    draw.text((num_x, y), val, font=big, fill=(255, 255, 255, 245))
    draw.text((unit_x, y + nh - px(38)), "touches", font=unit, fill=(*TEAL_RGB, 235))

    # Tempo meter — scale to 2.5 touches/s as the visual max.
    meter_y = y + nh + px(20)
    meter_w = px(300)
    meter_x = right_edge - meter_w
    draw.line((meter_x, meter_y, meter_x + meter_w, meter_y),
              fill=(255, 255, 255, 85), width=max(1, px(2)))
    scale_max = 2.5
    fill_w = int(meter_w * max(0.0, min(tempo / scale_max, 1.0)))
    if fill_w > 0:
        draw.line((meter_x, meter_y, meter_x + fill_w, meter_y),
                  fill=(*TEAL_RGB, 240), width=max(2, px(3)))

    sub_text = f"TEMPO  {tempo:.1f} /s"
    pbbox = draw.textbbox((0, 0), sub_text, font=sub_font)
    pw = pbbox[2] - pbbox[0]
    draw.text((right_edge - pw, meter_y + px(14)), sub_text, font=sub_font,
              fill=(255, 255, 255, 160))


def draw_speed_readout(base: Image.Image, speed: float, peak: float, scale_max: float = 25.0) -> None:
    """Borderless speed display in the top-right corner.

    Big thin numeral + km/h tag stacked to its right + a hairline meter
    underneath + a faint PEAK line. No card, no border.
    """
    W = base.size[0]
    pad = px(56)
    draw = ImageDraw.Draw(base)

    val = f"{speed:.1f}"
    big = _font(px(120), bold=False)
    unit = _font(px(22), bold=True)
    peak_font = _font(px(16), bold=True)

    # Right-aligned big number with a soft drop shadow for legibility on
    # bright backgrounds. The km/h tag is placed in the right-column,
    # stacked vertically next to the number, so it never overlaps.
    nbbox = draw.textbbox((0, 0), val, font=big)
    nw, nh = nbbox[2] - nbbox[0], nbbox[3] - nbbox[1]
    ubbox = draw.textbbox((0, 0), "km/h", font=unit)
    uw = ubbox[2] - ubbox[0]
    tag_gap = px(14)

    # Layout: [number]  [km/h]
    #         right-edge = W - pad
    right_edge = W - pad
    unit_x = right_edge - uw
    num_x = unit_x - tag_gap - nw
    y = pad

    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.text((num_x + px(3), y + px(3)), val, font=big, fill=(0, 0, 0, 180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(px(7)))
    base.alpha_composite(shadow)

    draw.text((num_x, y), val, font=big, fill=(255, 255, 255, 245))
    # km/h baseline-aligned with the number's bottom.
    draw.text((unit_x, y + nh - px(38)), "km/h", font=unit, fill=(*TEAL_RGB, 235))

    # Hairline meter under the readout — fills proportional to speed.
    meter_y = y + nh + px(20)
    meter_w = px(300)
    meter_x = right_edge - meter_w
    draw.line((meter_x, meter_y, meter_x + meter_w, meter_y),
              fill=(255, 255, 255, 85), width=max(1, px(2)))
    fill_w = int(meter_w * max(0.0, min(speed / scale_max, 1.0)))
    if fill_w > 0:
        draw.line((meter_x, meter_y, meter_x + fill_w, meter_y),
                  fill=(*TEAL_RGB, 240), width=max(2, px(3)))

    # PEAK subline.
    peak_text = f"PEAK  {peak:.1f}"
    pbbox = draw.textbbox((0, 0), peak_text, font=peak_font)
    pw = pbbox[2] - pbbox[0]
    draw.text((right_edge - pw, meter_y + px(14)), peak_text, font=peak_font,
              fill=(255, 255, 255, 160))


def draw_anchored_cue(
    base: Image.Image,
    header: str,
    body: str,
    alpha: float,
    anchor: tuple[int, int],
) -> None:
    """Floating coaching cue anchored to a point on the player.

    No box: a thin connector line from the anchor to a small text block
    (header in accent + body in white), with a soft drop shadow for
    legibility against any background. Alpha drives fade in / out.
    """
    if alpha <= 0.01:
        return
    W, H = base.size
    ax, ay = anchor

    header_font = _font(px(20), bold=True)
    body_font = _font(px(34), bold=True)

    measure = ImageDraw.Draw(base)
    tw = max(
        measure.textbbox((0, 0), header, font=header_font)[2],
        measure.textbbox((0, 0), body, font=body_font)[2],
    )
    th = px(34) + px(46) + px(8)
    margin = px(40)
    # Speed-readout no-fly zone (top-right corner). The cue must not start
    # inside this rectangle.
    speed_zone = (W - px(800), 0, W, px(300))

    def intersects_speed(rx: int, ry: int) -> bool:
        return not (
            rx + tw <= speed_zone[0]
            or rx >= speed_zone[2]
            or ry + th <= speed_zone[1]
            or ry >= speed_zone[3]
        )

    def in_bounds(rx: int, ry: int) -> bool:
        return (
            margin <= rx <= W - tw - margin
            and margin <= ry <= H - th - margin
        )

    # Candidate placements around the anchor, ranked by visual preference.
    dx = px(220)
    dy = px(180)
    candidates = [
        (ax + dx, ay - dy),         # right & above
        (ax - dx - tw, ay - dy),    # left  & above
        (ax + dx, ay + dy),         # right & below
        (ax - dx - tw, ay + dy),    # left  & below
    ]
    tx, ty = candidates[0]
    for cx, cy in candidates:
        cx_c = max(margin, min(cx, W - tw - margin))
        cy_c = max(margin, min(cy, H - th - margin))
        if in_bounds(cx_c, cy_c) and not intersects_speed(cx_c, cy_c):
            tx, ty = cx_c, cy_c
            break
    else:
        # Nothing fit; clamp the first candidate.
        tx = max(margin, min(tx, W - tw - margin))
        ty = max(margin, min(ty, H - th - margin))

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Strong dark backing for the text — render glyphs into an alpha-only
    # mask, blur it large, then composite as a near-opaque dark blob. This
    # is what makes the cue readable on bright domes / white kits.
    shadow_mask = Image.new("L", base.size, 0)
    sm = ImageDraw.Draw(shadow_mask)
    sm.text((tx, ty), header, font=header_font, fill=255)
    sm.text((tx, ty + px(34)), body, font=body_font, fill=255)
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(px(12)))
    # Boost the blurred mask so the dark blob is solid where text sits.
    shadow_mask = shadow_mask.point(lambda v: min(255, int(v * 2.6)))
    dark = Image.new("RGBA", base.size, (0, 0, 0, 230))
    layer.paste(dark, (0, 0), shadow_mask)

    # Connector hairline from anchor to the start of the text block. Dark
    # tone so it reads clearly against bright outdoor backgrounds; thin
    # bright halo on the anchor dot keeps it crisp.
    cx2, cy2 = tx - px(20), ty + px(30)
    dark = (12, 16, 22)
    d.line(
        [(ax, ay), (cx2, cy2)],
        fill=(*dark, 235),
        width=max(2, px(3)),
    )
    rr = max(3, px(6))
    d.ellipse((ax - rr, ay - rr, ax + rr, ay + rr), fill=(*dark, 245),
              outline=(255, 255, 255, 235), width=max(1, px(2)))

    # Text — header in orange, body in white.
    d.text((tx, ty), header, font=header_font, fill=(*ORANGE_RGB, 255))
    d.text((tx, ty + px(34)), body, font=body_font, fill=(255, 255, 255, 250))

    # Underline accent under the body.
    d.line(
        (tx, ty + px(34) + px(50), tx + px(90), ty + px(34) + px(50)),
        fill=(*ORANGE_RGB, 230),
        width=max(2, px(3)),
    )

    if alpha < 1.0:
        r, g, b, a = layer.split()
        a = a.point(lambda v: int(v * alpha))
        layer = Image.merge("RGBA", (r, g, b, a))
    base.alpha_composite(layer)


def draw_brand(base: Image.Image) -> None:
    """Tiny watermark, bottom-right."""
    W, H = base.size
    draw = ImageDraw.Draw(base)
    text = "SPORTS  /  PERF  ANALYSIS"
    f = _font(px(13), bold=True)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    x = W - px(32) - tw
    y = H - px(32) - px(14)
    # Subtle teal accent bar.
    draw.line((x - px(14), y + px(7), x - px(6), y + px(7)), fill=(*TEAL_RGB, 220), width=max(1, px(2)))
    draw.text((x, y), text, font=f, fill=(255, 255, 255, 180))


def draw_vignette(frame_bgr: np.ndarray) -> np.ndarray:
    """Subtle radial vignette to focus the eye and lift overlay contrast."""
    h, w = frame_bgr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    r = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
    mask = np.clip(1.0 - 0.32 * np.maximum(r - 0.55, 0) ** 1.6, 0.55, 1.0)
    return (frame_bgr.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)


# --- Skeleton & ball overlays (OpenCV — fast on the BGR frame) ---------

LEFT_EDGES = [(5, 7), (7, 9), (11, 13), (13, 15)]
RIGHT_EDGES = [(6, 8), (8, 10), (12, 14), (14, 16)]
CENTER_EDGES = [(5, 6), (11, 12), (5, 11), (6, 12)]


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, conf_thr: float = 0.30) -> None:
    """Thin glowing skeleton, left=teal, right=orange, center=white."""
    bone_outer = max(3, px(5))
    bone_inner = max(2, px(2))
    joint_outer = max(3, px(5))
    joint_inner = max(2, px(3))

    def bone(u, v, color):
        if kp[u, 2] < conf_thr or kp[v, 2] < conf_thr:
            return
        p1 = (int(kp[u, 0]), int(kp[u, 1]))
        p2 = (int(kp[v, 0]), int(kp[v, 1]))
        cv2.line(frame, p1, p2, (0, 0, 0), bone_outer, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, bone_inner, cv2.LINE_AA)

    for u, v in LEFT_EDGES:
        bone(u, v, TEAL)
    for u, v in RIGHT_EDGES:
        bone(u, v, ORANGE)
    for u, v in CENTER_EDGES:
        bone(u, v, (240, 240, 240))

    for i in range(17):
        if kp[i, 2] < conf_thr:
            continue
        c = (int(kp[i, 0]), int(kp[i, 1]))
        cv2.circle(frame, c, joint_outer, (0, 0, 0), -1, cv2.LINE_AA)
        col = TEAL if i in {5, 7, 9, 11, 13, 15} else ORANGE if i in {6, 8, 10, 12, 14, 16} else (255, 255, 255)
        cv2.circle(frame, c, joint_inner, col, -1, cv2.LINE_AA)


def draw_ball_trail(frame: np.ndarray, trail: list[tuple[float, float]]) -> None:
    """Glowing fading ball trail (BGR)."""
    if not trail:
        return
    overlay = frame.copy()
    n = len(trail)
    head_r = max(3, px(14))
    for i, (x, y) in enumerate(trail):
        age = (n - 1 - i) / max(n - 1, 1)
        r = int(head_r * (1.0 - age)) + max(1, px(2))
        cv2.circle(overlay, (int(x), int(y)), r, BALL_RGB[::-1], -1, cv2.LINE_AA)
    k = max(3, px(15)) | 1  # GaussianBlur kernel must be odd
    cv2.GaussianBlur(overlay, (k, k), 0, dst=overlay)
    cv2.addWeighted(overlay, 0.55, frame, 1.0, 0, dst=frame)
    x, y = trail[-1]
    cv2.circle(frame, (int(x), int(y)), max(3, px(6)), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(x), int(y)), max(2, px(4)), BALL_RGB[::-1], -1, cv2.LINE_AA)


# --- Coaching cues script ----------------------------------------------


CUE_SCRIPTS: dict[str, list[tuple[float, str, str, float]]] = {
    # (trigger_seconds, header, body, freeze_duration_seconds)
    "match": [
        (1.0, "FIRST TOUCH", "Soft control under pressure", 1.3),
        (5.0, "DRIVE", "Explode through the gap", 1.2),
        (9.0, "SCAN", "Eyes up — pick your spot", 1.2),
        (12.5, "STRIKE", "Plant foot — follow through", 1.4),
    ],
    "player_dribbling": [
        (1.2, "BALANCE", "Stay over the ball", 1.3),
        (4.5, "TOUCH", "Soft instep contact", 1.2),
        (8.0, "RHYTHM", "Find your tempo", 1.2),
        (11.2, "CONTROL", "Lock the ankle — eyes on the ball", 1.4),
    ],
}


def cue_schedule(name: str, fps: float) -> list[dict]:
    """Build the freeze-cue events for `name` at the source FPS."""
    script = CUE_SCRIPTS.get(name, CUE_SCRIPTS["match"])
    return [
        {"at": int(t * fps), "header": h, "body": b, "duration": d}
        for (t, h, b, d) in script
    ]


def freeze_alpha(k: int, total: int) -> float:
    """Fade envelope across `total` freeze frames: in 20%, hold 60%, out 20%."""
    t = k / max(total - 1, 1)
    if t < 0.20:
        return t / 0.20
    if t > 0.80:
        return max(0.0, (1.0 - t) / 0.20)
    return 1.0


def hero_anchor(s: dict, kp: np.ndarray | None) -> tuple[int, int]:
    """Pick an anchor point near the player's torso/chest."""
    if kp is not None and kp[5, 2] > 0.3 and kp[6, 2] > 0.3:
        ax = int((kp[5, 0] + kp[6, 0]) / 2)
        ay = int((kp[5, 1] + kp[6, 1]) / 2)
        return ax, ay
    x1, y1, x2, y2 = s["bbox"]
    return int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.25)


# --- Main render loop --------------------------------------------------


def render(src: Path, dst: Path, name: str) -> None:
    print(f"Detection pass on {src.name}…")
    frames, fps, n = detect_pass(src, (OUT_W, OUT_H))

    print("Selecting hero…")
    series = fill_gaps(hero_series(frames))
    present = sum(1 for s in series if s is not None)
    print(f"  hero present {present}/{len(frames)}")

    print("Smoothing keypoints…")
    kps = smooth_keypoints(series)

    use_juggles = name == "player_dribbling"
    if use_juggles:
        print("Counting juggles…")
        juggle_counts, juggle_tempo = count_juggles(frames, fps)
        print(f"  total juggles in clip: {juggle_counts[-1]}")
    else:
        print("Computing speed…")
        speeds, peaks = smooth_speed(series, fps)

    cues = cue_schedule(name, fps)
    cues_by_frame = {c["at"]: c for c in cues}

    print("Compositing frames…")
    # Try H.264 first (broad browser support); fall back to mp4v.
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"avc1"), fps, (OUT_W, OUT_H)
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H)
        )
        print("  (codec) avc1 unavailable; using mp4v")
    else:
        print("  (codec) avc1 / H.264")

    ball_trail: list[tuple[float, float]] = []
    cap = cv2.VideoCapture(str(src))

    def base_pil(frame_bgr: np.ndarray, src_i: int) -> Image.Image:
        f = frame_bgr.copy()
        if frames[src_i].ball:
            ball_trail.append(frames[src_i].ball)
            if len(ball_trail) > 28:
                ball_trail.pop(0)
        elif ball_trail:
            ball_trail.pop(0)
        draw_ball_trail(f, ball_trail)
        if kps[src_i] is not None:
            draw_skeleton(f, kps[src_i])
        pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("RGBA")
        if use_juggles:
            draw_juggle_readout(pil, juggle_counts[src_i], juggle_tempo[src_i])
        else:
            draw_speed_readout(pil, speeds[src_i], peaks[src_i])
        draw_brand(pil)
        return pil

    def pil_to_bgr(pil: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    out_count = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        frame = draw_vignette(frame)

        pil = base_pil(frame, i)
        writer.write(pil_to_bgr(pil))
        out_count += 1

        # Freeze-and-cue: at trigger frames, hold this frame for `duration`
        # seconds while the cue fades in/holds/out. Skip the cue (and the
        # freeze) when the hero isn't confidently identified — we never
        # want cues anchored to a defender or the keeper.
        if i in cues_by_frame and series[i] is not None and kps[i] is not None:
            cue = cues_by_frame[i]
            freeze_total = max(2, int(cue["duration"] * fps))
            anchor = hero_anchor(series[i], kps[i])
            for k in range(freeze_total):
                a = freeze_alpha(k, freeze_total)
                cue_frame = pil.copy()
                if a > 0:
                    draw_anchored_cue(cue_frame, cue["header"], cue["body"], a, anchor)
                writer.write(pil_to_bgr(cue_frame))
                out_count += 1

        if i % 25 == 0:
            print(f"  render src {i}/{n}  (out {out_count})")

    cap.release()
    writer.release()
    print(f"  total output frames: {out_count}  (= {out_count / fps:.1f} s)")
    print(f"Wrote {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="match",
                    choices=sorted(VIDEOS),
                    help="Video key from VIDEOS")
    args = ap.parse_args()

    src = ROOT / VIDEOS[args.name]
    dst = ROOT / f"{args.name}_annotated.mp4"
    render(src, dst, args.name)
