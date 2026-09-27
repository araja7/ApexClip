import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .scenes import Scene, detect_scenes, normalize_scenes

TRACKED_CLASSES = ("person", "sports ball")
CACHE_VERSION = 1
# Settings that change the scores; a cache made with different values is stale.
CACHE_KEY_FIELDS = ("model_path", "detection_conf", "stride", "top_k", "flow_width")


@dataclass
class Analysis:
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    scenes: List[Scene]
    scores: np.ndarray  # one excitement score per frame (diagonals/second)
    settings: Dict = field(default_factory=dict)
    video_signature: Dict = field(default_factory=dict)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "scenes": [list(s) for s in self.scenes],
            "scores": [round(float(s), 5) for s in self.scores],
            "settings": self.settings,
            "video_signature": self.video_signature,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "Analysis":
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != CACHE_VERSION:
            raise ValueError("analysis cache version mismatch")
        return cls(
            video_path=data["video_path"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            width=data["width"],
            height=data["height"],
            scenes=[tuple(s) for s in data["scenes"]],
            scores=np.asarray(data["scores"], dtype=float),
            settings=data.get("settings", {}),
            video_signature=data.get("video_signature", {}),
        )


def video_signature(video_path: str) -> Dict:
    st = os.stat(video_path)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def cache_settings(config: Config) -> Dict:
    return {k: getattr(config, k) for k in CACHE_KEY_FIELDS}


def load_cached(cache_path: str, video_path: str, config: Config) -> Optional[Analysis]:
    """Return the cached analysis if it exists and matches this video and these settings."""
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        cached = Analysis.load(cache_path)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    if cached.video_signature != video_signature(video_path):
        return None
    if cached.settings != cache_settings(config):
        return None
    return cached


# ---------------------------------------------------------------------------
# Motion math (pure functions, unit tested)
# ---------------------------------------------------------------------------

def estimate_camera_motion(
    prev_gray: np.ndarray, gray: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Estimate the global camera transform between two frames as a 2x3 similarity matrix.

    Tracks corner features (ideally on the background: pass a mask that blanks out
    players) with Lucas-Kanade optical flow and fits a rotation+scale+translation
    with RANSAC. Falls back to identity if there isn't enough texture.
    """
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask
    )
    if pts is None or len(pts) < 8:
        return identity
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None:
        return identity
    good = status.reshape(-1) == 1
    if good.sum() < 8:
        return identity
    matrix, _ = cv2.estimateAffinePartial2D(
        pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return identity if matrix is None else matrix


def scale_transform(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Convert a transform estimated on a frame resized by `scale` back to full resolution."""
    full = matrix.astype(float).copy()
    full[:, 2] /= scale
    return full


def compensated_displacement(
    prev_center: Tuple[float, float], curr_center: Tuple[float, float], camera: np.ndarray
) -> float:
    """How far an object moved in the scene after removing the camera's own movement.

    `camera` maps previous-frame coordinates to current-frame coordinates, so a
    stationary object is expected at camera @ prev_center.
    """
    px, py = prev_center
    expected = camera @ np.array([px, py, 1.0])
    return float(np.hypot(curr_center[0] - expected[0], curr_center[1] - expected[1]))


def frame_score(speeds: List[float], top_k: int) -> float:
    """Mean of the K fastest tracks. Ignoring slow tracks keeps idle players, refs and
    fans in the stands from diluting a play where a handful of players are sprinting."""
    if not speeds:
        return 0.0
    fastest = sorted(speeds, reverse=True)[:top_k]
    return float(np.mean(fastest))


def interpolate_scores(indices: List[int], values: List[float], total_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(0)
    if not indices:
        return np.zeros(total_frames)
    return np.interp(np.arange(total_frames), indices, values)


# ---------------------------------------------------------------------------
# Main analysis pass
# ---------------------------------------------------------------------------

def _reset_tracker(model):
    predictor = getattr(model, "predictor", None)
    if predictor is not None and getattr(predictor, "trackers", None):
        for tracker in predictor.trackers:
            tracker.reset()


def _draw_diagnostics(frame, detections, score, scene_idx, frame_idx):
    palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    for (x1, y1, x2, y2), track_id, name, speed in detections:
        color = palette[track_id % len(palette)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {name} {speed:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    bar = int(min(score / 0.5, 1.0) * 300)
    cv2.rectangle(frame, (10, 10), (310, 40), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 10), (10 + bar, 40), (0, 140, 255), -1)
    cv2.putText(
        frame, f"score {score:.3f}  scene {scene_idx}  frame {frame_idx}",
        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def analyze(
    video_path: str,
    config: Config,
    diagnostic_path: Optional[str] = None,
    log=print,
) -> Analysis:
    from ultralytics import YOLO  # imported lazily: slow import, not needed for cached runs

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diagonal = float(np.hypot(width, height))

    log("Detecting camera cuts...")
    scenes = detect_scenes(video_path, reported_frames)
    log(f"  found {len(scenes)} scene(s)")

    log(f"Loading model {config.model_path}...")
    model = YOLO(config.model_path)
    class_ids = [i for i, name in model.names.items() if name in TRACKED_CLASSES]

    writer = None
    if diagnostic_path:
        os.makedirs(os.path.dirname(os.path.abspath(diagnostic_path)), exist_ok=True)
        writer = cv2.VideoWriter(
            diagnostic_path, cv2.VideoWriter_fourcc(*"mp4v"),
            fps / config.stride, (width, height),
        )

    flow_scale = min(1.0, config.flow_width / width) if width else 1.0
    scene_idx = 0
    track_memory: Dict[int, Tuple[float, float]] = {}
    prev_gray = None
    prev_boxes: List[Tuple[int, int, int, int]] = []
    last_idx = None
    score_indices: List[int] = []
    score_values: List[float] = []

    log("Analyzing motion...")
    started = time.time()
    next_report = 0.0
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Camera cut: previous positions and the tracker's IDs no longer mean anything.
        while scene_idx < len(scenes) - 1 and frame_idx >= scenes[scene_idx][1]:
            scene_idx += 1
            track_memory.clear()
            prev_gray = None
            prev_boxes = []
            last_idx = None
            _reset_tracker(model)

        if frame_idx % config.stride != 0:
            continue

        small = cv2.resize(frame, None, fx=flow_scale, fy=flow_scale) if flow_scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        camera = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        if prev_gray is not None:
            mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
            for x1, y1, x2, y2 in prev_boxes:
                mask[int(y1 * flow_scale):int(y2 * flow_scale) + 1,
                     int(x1 * flow_scale):int(x2 * flow_scale) + 1] = 0
            camera = scale_transform(estimate_camera_motion(prev_gray, gray, mask), flow_scale)

        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=config.detection_conf,
            classes=class_ids, device=config.device, verbose=False,
        )[0]

        speeds: List[float] = []
        detections = []
        boxes_now: List[Tuple[int, int, int, int]] = []
        new_memory: Dict[int, Tuple[float, float]] = {}
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            gap = frame_idx - last_idx if last_idx is not None else None
            for (x1, y1, x2, y2), track_id, cls in zip(boxes, track_ids, classes):
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                speed = 0.0
                if gap and track_id in track_memory:
                    pixels = compensated_displacement(track_memory[track_id], center, camera)
                    speed = pixels / diagonal * fps / gap  # frame-diagonals per second
                    speeds.append(speed)
                new_memory[track_id] = center
                boxes_now.append((x1, y1, x2, y2))
                detections.append(((x1, y1, x2, y2), int(track_id), model.names[cls], speed))
        track_memory = new_memory

        score = frame_score(speeds, config.top_k)
        score_indices.append(frame_idx)
        score_values.append(score)

        if writer is not None:
            _draw_diagnostics(frame, detections, score, scene_idx, frame_idx)
            writer.write(frame)

        prev_gray = gray
        prev_boxes = boxes_now
        last_idx = frame_idx

        if reported_frames > 0:
            progress = frame_idx / reported_frames
            if progress >= next_report:
                elapsed = time.time() - started
                rate = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                log(f"  {progress * 100:5.1f}%  frame {frame_idx}/{reported_frames}  ({rate:.1f} frames/s)")
                next_report += 0.05

    cap.release()
    if writer is not None:
        writer.release()

    total_frames = frame_idx + 1
    if total_frames != reported_frames:
        scenes = normalize_scenes(scenes, total_frames)

    return Analysis(
        video_path=os.path.abspath(video_path),
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        scenes=scenes,
        scores=interpolate_scores(score_indices, score_values, total_frames),
        settings=cache_settings(config),
        video_signature=video_signature(video_path),
    )
