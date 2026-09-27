"""Turn a per-frame excitement curve into a list of highlight segments.

Everything here is pure (arrays in, segments out) so it can be re-run instantly on a
cached analysis while tuning, and unit tested without any video.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .scenes import Scene, scene_index_for_frame


@dataclass
class Event:
    start: int  # first frame above threshold
    end: int  # exclusive
    peak_frame: int
    peak_score: float


@dataclass
class Segment:
    start: int  # frame, inclusive
    end: int  # frame, exclusive
    peak_frame: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def smooth(scores: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that doesn't sag at the edges of the video."""
    scores = np.asarray(scores, dtype=float)
    if window <= 1 or len(scores) == 0:
        return scores.copy()
    kernel = np.ones(window)
    total = np.convolve(scores, kernel, mode="same")
    counts = np.convolve(np.ones(len(scores)), kernel, mode="same")
    return total / counts


def compute_threshold(smoothed: np.ndarray, config: Config) -> float:
    if config.threshold is not None:
        return config.threshold
    if len(smoothed) == 0:
        return config.min_threshold
    return max(float(np.percentile(smoothed, config.percentile)), config.min_threshold)


def find_events(
    smoothed: np.ndarray, threshold: float, min_length: int, release: Optional[float] = None
) -> List[Event]:
    """Find plays using hysteresis.

    A play is a run of frames above `release` (default: the threshold itself) that
    reaches `threshold` somewhere and lasts at least `min_length` frames. Using a lower
    release level keeps one sustained play in one piece even when its motion hovers
    around the trigger threshold, and captures the build-up and wind-down.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    release = threshold if release is None else min(release, threshold)
    above = smoothed > release
    events = []
    i, n = 0, len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        peak = i + int(np.argmax(smoothed[i:j]))
        if smoothed[peak] > threshold and j - i >= max(min_length, 1):
            events.append(Event(i, j, peak, float(smoothed[peak])))
        i = j
    return events


def expand_event(
    event: Event, scenes: List[Scene], pre_roll: int, post_roll: int, max_length: int
) -> Segment:
    """Add pre/post-roll around an event without crossing a camera cut.

    The pre-roll shows the setup (the snap, the drive to the basket); clamping to the
    scene keeps us from opening the clip on an unrelated shot.
    """
    scene_start, scene_end = scenes[scene_index_for_frame(scenes, event.peak_frame)]
    start = max(event.start - pre_roll, scene_start)
    end = min(event.end + post_roll, scene_end)
    if max_length > 0 and end - start > max_length:
        # Keep as much pre-roll as fits, but the window must still contain the peak.
        earliest = max(start, event.peak_frame - max_length + 1)
        latest = end - max_length
        start = int(np.clip(event.start - pre_roll, earliest, latest))
        end = start + max_length
    return Segment(start, end, event.peak_frame, event.peak_score)


def merge_segments(segments: List[Segment], max_gap: int) -> List[Segment]:
    """Merge segments that overlap or are separated by a short gap, so no frame is
    shown twice and back-to-back bursts of one play become a single clip."""
    merged: List[Segment] = []
    for seg in sorted(segments, key=lambda s: s.start):
        if merged and seg.start - merged[-1].end <= max_gap:
            last = merged[-1]
            best = last if last.score >= seg.score else seg
            merged[-1] = Segment(last.start, max(last.end, seg.end), best.peak_frame, best.score)
        else:
            merged.append(Segment(seg.start, seg.end, seg.peak_frame, seg.score))
    return merged


def trim_around_peak(seg: Segment, length: int) -> Segment:
    if length >= seg.length:
        return seg
    start = int(np.clip(seg.peak_frame - length // 2, seg.start, seg.end - length))
    return Segment(start, start + length, seg.peak_frame, seg.score)


def select_within_budget(segments: List[Segment], budget: Optional[int]) -> List[Segment]:
    """Keep the highest-scoring segments that fit in the budget, in chronological order.

    A segment that doesn't quite fit is trimmed around its peak (as long as at least
    half of it survives) rather than dropped in favor of a weaker play.
    """
    if budget is None:
        return sorted(segments, key=lambda s: s.start)
    chosen: List[Segment] = []
    used = 0
    for seg in sorted(segments, key=lambda s: s.score, reverse=True):
        remaining = budget - used
        if seg.length <= remaining:
            chosen.append(seg)
        elif remaining > 0 and remaining * 2 >= seg.length:
            chosen.append(trim_around_peak(seg, remaining))
        else:
            continue
        used += chosen[-1].length
    if not chosen and segments and budget > 0:
        # Even the best segment is far longer than the whole budget.
        chosen = [trim_around_peak(max(segments, key=lambda s: s.score), budget)]
    return sorted(chosen, key=lambda s: s.start)


def plan_highlights(
    scores: np.ndarray, scenes: List[Scene], fps: float, config: Config
) -> Tuple[List[Segment], float, np.ndarray]:
    """Full selection pipeline. Returns (segments, threshold used, smoothed curve)."""
    def frames(seconds: float) -> int:
        return int(round(seconds * fps))

    smoothed = smooth(scores, frames(config.smoothing_seconds))
    threshold = compute_threshold(smoothed, config)
    events = find_events(
        smoothed, threshold, frames(config.min_event_seconds), threshold * config.release_ratio
    )
    segments = [
        expand_event(
            e, scenes, frames(config.pre_roll_seconds), frames(config.post_roll_seconds),
            frames(config.max_clip_seconds),
        )
        for e in events
    ]
    segments = merge_segments(segments, frames(config.merge_gap_seconds))
    budget = None if config.max_duration_seconds is None else frames(config.max_duration_seconds)
    return select_within_budget(segments, budget), threshold, smoothed
