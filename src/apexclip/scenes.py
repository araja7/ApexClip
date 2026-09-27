from typing import List, Tuple

from scenedetect import ContentDetector, detect

Scene = Tuple[int, int]  # (start_frame, end_frame), end exclusive


def detect_scenes(video_path: str, total_frames: int) -> List[Scene]:
    """Return camera-cut boundaries as contiguous frame ranges covering the whole video."""
    scene_list = detect(video_path, ContentDetector())
    scenes = [(start.frame_num, end.frame_num) for start, end in scene_list]
    return normalize_scenes(scenes, total_frames)


def normalize_scenes(scenes: List[Scene], total_frames: int) -> List[Scene]:
    """Guarantee scenes are sorted, non-empty, and span [0, total_frames)."""
    if total_frames <= 0:
        return []
    starts = sorted({s for s, _ in scenes if 0 < s < total_frames})
    bounds = [0] + starts + [total_frames]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def scene_index_for_frame(scenes: List[Scene], frame: int) -> int:
    for i, (start, end) in enumerate(scenes):
        if start <= frame < end:
            return i
    return len(scenes) - 1
