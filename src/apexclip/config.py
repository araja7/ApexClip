from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    # --- Analysis (expensive pass: YOLO + tracking) ---
    model_path: str = "yolov8n.pt"
    detection_conf: float = 0.15
    device: Optional[str] = None  # None lets ultralytics choose; "mps", "cuda:0", "cpu"
    stride: int = 1  # analyze every Nth frame; skipped frames are interpolated
    top_k: int = 5  # a frame's score is the mean speed of its K fastest tracks
    flow_width: int = 640  # frames are downscaled to this width for camera-motion estimation

    # --- Selection (cheap pass: runs on cached scores) ---
    smoothing_seconds: float = 0.5
    threshold: Optional[float] = None  # absolute score threshold; None = use percentile
    percentile: float = 90.0
    min_threshold: float = 0.05  # floor for the adaptive threshold (diagonals/second)
    release_ratio: float = 0.6  # a play continues while motion stays above threshold * this
    min_event_seconds: float = 0.4  # a play must last at least this long to count
    pre_roll_seconds: float = 6.0
    post_roll_seconds: float = 3.0
    merge_gap_seconds: float = 1.0
    max_clip_seconds: float = 20.0
    max_duration_seconds: Optional[float] = 90.0  # total reel length budget; None = no limit

    # --- Render ---
    crf: int = 20
    preset: str = "veryfast"

    def to_dict(self):
        return asdict(self)
