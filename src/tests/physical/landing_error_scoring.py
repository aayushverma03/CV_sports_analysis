"""Landing Error Scoring System (LESS) — subset score.

Single side-on camera limits us to the sagittal-plane subset of Padua's
17-item LESS rubric. v1 ships an 8-item subset:

  1. stiff_knee_at_ic           - knee flexion < 30 deg at initial contact
  2. stiff_hip_at_ic             - hip  flexion < 30 deg at IC
  3. upright_trunk_at_ic         - trunk flexion < 20 deg at IC
  4. insufficient_knee_disp      - peak knee flex - IC knee flex < 45 deg
  5. insufficient_hip_disp       - peak hip  flex - IC hip  flex < 30 deg
  6. insufficient_trunk_disp     - peak trunk flex - IC trunk flex < 10 deg
  7. asymmetric_initial_contact  - |L - R ankle y| at IC > 5% bbox-h
  8. overall_impression          - 4+ of items 1-7 flagged

Sum = `less_error_score` (0-8, lower = better technique). Frontal-plane
items (knee valgus, lateral trunk lean, stance width) are deferred to a
two-camera pipeline.

Detection mirrors Drop Jump: athlete stands on a box, drops, lands. The
state machine locks the box level over the first N frames, watches for
ankle drop, then plateaus at first ground contact (= IC). Peak flexion
is found by scanning a window of frames after IC.

Pose backend: `pose_biomech` (RTMPose-x). Joint angles need precise
keypoints — the YOLO-pose default is too noisy for sub-30-deg cutoffs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.core.annotation.overlays import (
    draw_bbox,
    draw_hud,
    draw_skeleton,
    event_flash,
    render_endcard,
)
from src.core.detection.player_detector import detect_players
from src.core.pose.estimator import PoseDetection, create_pose_estimator
from src.core.utils.geometry import angle_3pt
from src.core.utils.video_io import frame_iter, video_info
from src.scoring.grade import format_band
from src.tests.base import (
    AnalysisDiagnostics,
    AnalysisResult,
    AthleteProfile,
    BaseTest,
    MetricValue,
    ProtocolError,
    score_test,
)

# --- Tunables ----------------------------------------------------------

_ON_BOX_FRAMES = 15
_DROP_THRESHOLD_FRAC = 0.05
_LANDING_PLATEAU_FRAMES = 3
_LANDING_VELOCITY_FRAC = 0.02

_POSE_CONF_MIN = 0.30
_POST_IC_PEAK_WINDOW_FRAMES = 30   # search window after IC for max flexion

# LESS error thresholds (sagittal-plane subset, Padua 2009).
_STIFF_KNEE_DEG = 30.0
_STIFF_HIP_DEG = 30.0
_STIFF_TRUNK_DEG = 20.0
_KNEE_DISPLACEMENT_DEG = 45.0
_HIP_DISPLACEMENT_DEG = 30.0
_TRUNK_DISPLACEMENT_DEG = 10.0
_ASYMMETRY_FRAC = 0.05
_OVERALL_IMPRESSION_THRESHOLD = 4

_ENDCARD_HOLD_S = 2.5


_State = Literal["on_box", "dropping", "contact_1", "done"]


# --- Detector ---------------------------------------------------------


@dataclass
class _LessDetector:
    """Streaming state machine — locates the initial-contact frame."""

    state: _State = "on_box"
    box_y: float | None = None
    ground_y: float | None = None
    step_off_frame: int | None = None
    initial_contact_frame: int | None = None

    _box_buffer: list[float] = field(default_factory=list)
    _last_y: float | None = None
    _stable_count: int = 0
    _provisional_landing_frame: int | None = None

    def update(
        self,
        frame_idx: int,
        ankle_y: float | None,
        bbox_h: float | None,
    ) -> None:
        if ankle_y is None or bbox_h is None:
            self._last_y = None
            return

        if self.state == "on_box":
            self._box_buffer.append(ankle_y)
            if len(self._box_buffer) >= _ON_BOX_FRAMES:
                self.box_y = float(np.median(self._box_buffer))
            if (
                self.box_y is not None
                and ankle_y > self.box_y + _DROP_THRESHOLD_FRAC * bbox_h
            ):
                self.state = "dropping"
                self.step_off_frame = frame_idx

        elif self.state == "dropping":
            if self._last_y is not None:
                delta = abs(ankle_y - self._last_y)
                if delta < _LANDING_VELOCITY_FRAC * bbox_h:
                    if self._stable_count == 0:
                        self._provisional_landing_frame = frame_idx - 1
                    self._stable_count += 1
                else:
                    self._stable_count = 0
                    self._provisional_landing_frame = None
            if self._stable_count >= _LANDING_PLATEAU_FRAMES:
                self.state = "contact_1"
                self.initial_contact_frame = self._provisional_landing_frame
                self.ground_y = ankle_y

        elif self.state == "contact_1":
            # Stay here for the post-IC peak-flexion window; pipeline
            # decides when we're done.
            pass

        self._last_y = ankle_y


# --- Joint angle helpers ----------------------------------------------


def _knee_flexion_deg(pose: PoseDetection | None) -> float | None:
    """Knee flexion = 180 - interior(hip, knee, ankle). 0 = straight,
    larger = more bent. Averaged across legs that pass the confidence
    gate. Returns None if no leg is fully confident."""
    if pose is None:
        return None
    flexions: list[float] = []
    for side in ("left", "right"):
        if min(
            pose.confidence_of(f"{side}_hip"),
            pose.confidence_of(f"{side}_knee"),
            pose.confidence_of(f"{side}_ankle"),
        ) < _POSE_CONF_MIN:
            continue
        try:
            interior = angle_3pt(
                pose.position(f"{side}_hip"),
                pose.position(f"{side}_knee"),
                pose.position(f"{side}_ankle"),
            )
        except ValueError:
            continue
        flexions.append(180.0 - interior)
    if not flexions:
        return None
    return float(np.mean(flexions))


def _hip_flexion_deg(pose: PoseDetection | None) -> float | None:
    """Hip flexion = 180 - interior(shoulder, hip, knee). 0 = upright,
    larger = more bent at the hip."""
    if pose is None:
        return None
    flexions: list[float] = []
    for side in ("left", "right"):
        if min(
            pose.confidence_of(f"{side}_shoulder"),
            pose.confidence_of(f"{side}_hip"),
            pose.confidence_of(f"{side}_knee"),
        ) < _POSE_CONF_MIN:
            continue
        try:
            interior = angle_3pt(
                pose.position(f"{side}_shoulder"),
                pose.position(f"{side}_hip"),
                pose.position(f"{side}_knee"),
            )
        except ValueError:
            continue
        flexions.append(180.0 - interior)
    if not flexions:
        return None
    return float(np.mean(flexions))


def _trunk_flexion_deg(pose: PoseDetection | None) -> float | None:
    """Trunk forward flexion: angle between mid-hip -> mid-shoulder
    vector and image vertical (-y axis). 0 = upright, larger = more
    forward lean."""
    if pose is None:
        return None
    needed = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if any(pose.confidence_of(k) < _POSE_CONF_MIN for k in needed):
        return None
    hip = (
        np.asarray(pose.position("left_hip"), dtype=float)
        + np.asarray(pose.position("right_hip"), dtype=float)
    ) / 2.0
    sh = (
        np.asarray(pose.position("left_shoulder"), dtype=float)
        + np.asarray(pose.position("right_shoulder"), dtype=float)
    ) / 2.0
    v = sh - hip
    n = float(np.linalg.norm(v))
    if n == 0:
        return None
    cos_a = float(np.clip(-v[1] / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _ankle_asymmetry_frac(
    pose: PoseDetection | None, bbox_h: float | None,
) -> float | None:
    """|left_ankle_y - right_ankle_y| / bbox_h. Higher = more asymmetric
    initial contact (one foot lands first)."""
    if pose is None or bbox_h is None or bbox_h <= 0:
        return None
    if min(
        pose.confidence_of("left_ankle"),
        pose.confidence_of("right_ankle"),
    ) < _POSE_CONF_MIN:
        return None
    ly = float(pose.position("left_ankle")[1])
    ry = float(pose.position("right_ankle")[1])
    return abs(ly - ry) / bbox_h


# --- Scoring ----------------------------------------------------------


@dataclass(frozen=True)
class _LessItems:
    stiff_knee_at_ic: bool
    stiff_hip_at_ic: bool
    upright_trunk_at_ic: bool
    insufficient_knee_disp: bool
    insufficient_hip_disp: bool
    insufficient_trunk_disp: bool
    asymmetric_initial_contact: bool
    overall_impression: bool

    @property
    def total(self) -> int:
        return sum(
            int(v) for v in (
                self.stiff_knee_at_ic, self.stiff_hip_at_ic,
                self.upright_trunk_at_ic, self.insufficient_knee_disp,
                self.insufficient_hip_disp, self.insufficient_trunk_disp,
                self.asymmetric_initial_contact, self.overall_impression,
            )
        )


def _score_landing_errors(
    *,
    knee_flex_ic: float,
    hip_flex_ic: float,
    trunk_flex_ic: float,
    knee_flex_peak: float,
    hip_flex_peak: float,
    trunk_flex_peak: float,
    ankle_asymmetry_frac: float,
) -> _LessItems:
    """Apply the 8-item subset rubric. Inputs are degrees; the asymmetry
    input is already normalised to bbox-h. Pure function — drives unit
    tests."""
    stiff_knee = knee_flex_ic < _STIFF_KNEE_DEG
    stiff_hip = hip_flex_ic < _STIFF_HIP_DEG
    upright_trunk = trunk_flex_ic < _STIFF_TRUNK_DEG
    insuff_knee = (knee_flex_peak - knee_flex_ic) < _KNEE_DISPLACEMENT_DEG
    insuff_hip = (hip_flex_peak - hip_flex_ic) < _HIP_DISPLACEMENT_DEG
    insuff_trunk = (
        trunk_flex_peak - trunk_flex_ic
    ) < _TRUNK_DISPLACEMENT_DEG
    asymmetric = ankle_asymmetry_frac > _ASYMMETRY_FRAC
    flagged_count = sum(
        int(b) for b in (
            stiff_knee, stiff_hip, upright_trunk, insuff_knee,
            insuff_hip, insuff_trunk, asymmetric,
        )
    )
    overall = flagged_count >= _OVERALL_IMPRESSION_THRESHOLD
    return _LessItems(
        stiff_knee_at_ic=stiff_knee,
        stiff_hip_at_ic=stiff_hip,
        upright_trunk_at_ic=upright_trunk,
        insufficient_knee_disp=insuff_knee,
        insufficient_hip_disp=insuff_hip,
        insufficient_trunk_disp=insuff_trunk,
        asymmetric_initial_contact=asymmetric,
        overall_impression=overall,
    )


# --- Pipeline ---------------------------------------------------------


class LandingErrorScoringTest(BaseTest):
    """LESS subset: 8-item sagittal-plane scoring on a drop-and-land."""

    test_id = "landing-error-scoring-system"

    def __init__(self) -> None:
        self._pose = create_pose_estimator("pose_biomech")

    def run(
        self,
        video_path: Path,
        athlete: AthleteProfile,
        output_dir: Path,
    ) -> AnalysisResult:
        info = video_info(video_path)
        fps = info.fps
        out_path = output_dir / f"{self.test_id}.mp4"

        # === PASS 1: detect+pose, cache per-frame state ===
        detector = _LessDetector()
        pose_by_frame: dict[int, PoseDetection] = {}
        bbox_by_frame: dict[int, np.ndarray] = {}
        bbox_h_by_frame: dict[int, float] = {}
        n_frames = 0
        n_low_conf = 0

        for frame in frame_iter(video_path):
            n_frames += 1
            dets = detect_players(frame.image)
            bbox = dets[0].bbox_xyxy if dets else None
            pose = (
                self._pose.estimate_bbox(frame.image, bbox)
                if bbox is not None else None
            )
            if pose is not None:
                pose_by_frame[frame.idx] = pose
                if pose.mean_confidence < _POSE_CONF_MIN:
                    n_low_conf += 1
            if bbox is not None:
                bbox_by_frame[frame.idx] = bbox
                bbox_h_by_frame[frame.idx] = float(bbox[3] - bbox[1])

            ankle_y, bbox_h = _ankle_features(pose, bbox)
            detector.update(frame.idx, ankle_y, bbox_h)

        ic_frame = detector.initial_contact_frame
        if ic_frame is None:
            raise ProtocolError(
                "no initial-contact frame detected — athlete may not have "
                "stepped off the box, or pose tracking lost the ankles"
            )

        # === Sample IC pose + scan window for peak flexion ===
        ic_pose = pose_by_frame.get(ic_frame)
        ic_bbox_h = bbox_h_by_frame.get(ic_frame)
        if ic_pose is None or ic_bbox_h is None:
            raise ProtocolError(
                "pose at initial-contact frame is missing; cannot compute "
                "LESS items"
            )

        knee_flex_ic = _knee_flexion_deg(ic_pose)
        hip_flex_ic = _hip_flexion_deg(ic_pose)
        trunk_flex_ic = _trunk_flexion_deg(ic_pose)
        asymmetry = _ankle_asymmetry_frac(ic_pose, ic_bbox_h)
        if (knee_flex_ic is None or hip_flex_ic is None
                or trunk_flex_ic is None or asymmetry is None):
            raise ProtocolError(
                "required keypoints below confidence threshold at IC; "
                "cannot compute LESS items"
            )

        peak_knee, peak_hip, peak_trunk = _scan_peak_flexion(
            pose_by_frame=pose_by_frame,
            ic_frame=ic_frame,
            window=_POST_IC_PEAK_WINDOW_FRAMES,
        )
        # Peak should never be below IC; if window has no usable frames
        # (low confidence), fall back to IC values (yields zero
        # displacement -> the displacement items will flag).
        peak_knee = max(peak_knee, knee_flex_ic)
        peak_hip = max(peak_hip, hip_flex_ic)
        peak_trunk = max(peak_trunk, trunk_flex_ic)

        items = _score_landing_errors(
            knee_flex_ic=knee_flex_ic,
            hip_flex_ic=hip_flex_ic,
            trunk_flex_ic=trunk_flex_ic,
            knee_flex_peak=peak_knee,
            hip_flex_peak=peak_hip,
            trunk_flex_peak=peak_trunk,
            ankle_asymmetry_frac=asymmetry,
        )

        metrics: dict[str, MetricValue] = {
            "less_error_score": MetricValue(raw=float(items.total), unit="count"),
            "knee_flexion_at_ic_deg": MetricValue(raw=knee_flex_ic, unit="deg"),
            "hip_flexion_at_ic_deg": MetricValue(raw=hip_flex_ic, unit="deg"),
            "trunk_flexion_at_ic_deg": MetricValue(raw=trunk_flex_ic, unit="deg"),
            "knee_flexion_displacement_deg": MetricValue(
                raw=peak_knee - knee_flex_ic, unit="deg",
            ),
            "hip_flexion_displacement_deg": MetricValue(
                raw=peak_hip - hip_flex_ic, unit="deg",
            ),
            "trunk_flexion_displacement_deg": MetricValue(
                raw=peak_trunk - trunk_flex_ic, unit="deg",
            ),
            "ankle_asymmetry_at_ic_pct": MetricValue(
                raw=asymmetry * 100.0, unit="pct",
            ),
        }
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 2: render ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        try:
            for frame in frame_iter(video_path):
                img = frame.image
                bbox = bbox_by_frame.get(frame.idx)
                pose = pose_by_frame.get(frame.idx)
                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                if bbox is not None:
                    draw_bbox(img, bbox)
                draw_hud(img, _hud_fields(
                    frame_idx=frame.idx, fps=fps, ic_frame=ic_frame,
                    items=items,
                ))
                if frame.idx == ic_frame and bbox is not None:
                    event_flash(img, bbox)
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.2f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="LESS (subset)",
                athlete=f"{athlete.gender} age {athlete.age}",
                metric_rows=endcard_rows,
                test_score=int(round(test_score.score)),
                band=format_band(test_score.band),
                size=(info.width, info.height),
            )
            for _ in range(int(fps * _ENDCARD_HOLD_S)):
                writer.write(endcard)
        finally:
            writer.release()

        return AnalysisResult(
            test_id=self.test_id,
            athlete=athlete,
            metrics=metrics,
            scores=scores,
            test_score=test_score,
            annotated_video_path=out_path,
            diagnostics=AnalysisDiagnostics(
                fps_input=fps, duration_s=n_frames / fps if fps > 0 else 0.0,
            ),
        )


# --- helpers ----------------------------------------------------------


def _ankle_features(
    pose: PoseDetection | None, bbox: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Min ankle-y (highest in image) + bbox-h, or (None, None) if
    neither ankle is confident."""
    if pose is None or bbox is None:
        return None, None
    candidates: list[float] = []
    for kp in ("left_ankle", "right_ankle"):
        if pose.confidence_of(kp) >= _POSE_CONF_MIN:
            candidates.append(float(pose.position(kp)[1]))
    if not candidates:
        return None, None
    return min(candidates), float(bbox[3] - bbox[1])


def _scan_peak_flexion(
    *,
    pose_by_frame: dict[int, PoseDetection],
    ic_frame: int,
    window: int,
) -> tuple[float, float, float]:
    """Return (peak_knee, peak_hip, peak_trunk) flexion in degrees over
    the [ic_frame, ic_frame + window] frames. 0.0 if no usable poses."""
    peak_knee = 0.0
    peak_hip = 0.0
    peak_trunk = 0.0
    for fi in range(ic_frame, ic_frame + window + 1):
        pose = pose_by_frame.get(fi)
        if pose is None:
            continue
        kf = _knee_flexion_deg(pose)
        hf = _hip_flexion_deg(pose)
        tf = _trunk_flexion_deg(pose)
        if kf is not None:
            peak_knee = max(peak_knee, kf)
        if hf is not None:
            peak_hip = max(peak_hip, hf)
        if tf is not None:
            peak_trunk = max(peak_trunk, tf)
    return peak_knee, peak_hip, peak_trunk


def _hud_fields(
    *,
    frame_idx: int,
    fps: float,
    ic_frame: int,
    items: _LessItems,
) -> dict[str, str]:
    if frame_idx < ic_frame:
        return {"phase": "drop", "errors": "-"}
    if frame_idx == ic_frame:
        return {"phase": "contact", "errors": str(items.total)}
    return {"phase": "absorb", "errors": str(items.total)}
