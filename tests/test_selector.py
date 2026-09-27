import numpy as np
import pytest

from apexclip.config import Config
from apexclip.selector import (
    Event,
    Segment,
    compute_threshold,
    expand_event,
    find_events,
    merge_segments,
    plan_highlights,
    select_within_budget,
    smooth,
)

FPS = 10.0


class TestSmooth:
    def test_constant_signal_unchanged_including_edges(self):
        out = smooth(np.full(20, 3.0), 5)
        assert np.allclose(out, 3.0)

    def test_single_spike_is_spread_out(self):
        scores = np.zeros(21)
        scores[10] = 5.0
        out = smooth(scores, 5)
        assert out[10] == pytest.approx(1.0)
        assert out[8] == pytest.approx(1.0) and out[12] == pytest.approx(1.0)
        assert out[7] == 0 and out[13] == 0

    def test_window_of_one_is_identity(self):
        scores = np.array([1.0, 4.0, 2.0])
        assert np.array_equal(smooth(scores, 1), scores)

    def test_empty(self):
        assert len(smooth(np.array([]), 5)) == 0


class TestThreshold:
    def test_absolute_threshold_wins(self):
        assert compute_threshold(np.arange(100.0), Config(threshold=0.7)) == 0.7

    def test_percentile(self):
        t = compute_threshold(np.arange(101.0), Config(percentile=90, min_threshold=0))
        assert t == pytest.approx(90.0)

    def test_floor_prevents_picking_noise_in_a_dead_video(self):
        t = compute_threshold(np.full(100, 0.001), Config(min_threshold=0.05))
        assert t == 0.05


class TestFindEvents:
    def test_finds_runs_and_peaks(self):
        s = np.array([0, 0, 2, 3, 5, 2, 0, 0, 4, 4, 0], dtype=float)
        events = find_events(s, 1.0, 1)
        assert [(e.start, e.end, e.peak_frame) for e in events] == [(2, 6, 4), (8, 10, 8)]
        assert events[0].peak_score == 5

    def test_min_length_drops_short_blips(self):
        s = np.array([0, 5, 0, 5, 5, 5, 0], dtype=float)
        events = find_events(s, 1.0, 3)
        assert [(e.start, e.end) for e in events] == [(3, 6)]

    def test_hysteresis_keeps_a_play_that_hovers_around_threshold_in_one_piece(self):
        s = np.array([0, 0.7, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 0, 0], dtype=float)
        assert len(find_events(s, 1.0, 1)) == 3  # fragmented without hysteresis
        events = find_events(s, 1.0, 1, release=0.6)
        assert [(e.start, e.end, e.peak_frame) for e in events] == [(1, 8, 6)]

    def test_hysteresis_run_must_reach_threshold(self):
        s = np.array([0, 0.8, 0.9, 0.8, 0, 0.7, 1.5, 0.7, 0], dtype=float)
        events = find_events(s, 1.0, 1, release=0.6)
        assert [(e.start, e.end) for e in events] == [(5, 8)]

    def test_run_touching_end_of_video(self):
        events = find_events(np.array([0, 0, 2, 2], dtype=float), 1.0, 1)
        assert [(e.start, e.end) for e in events] == [(2, 4)]


class TestExpandEvent:
    scenes = [(0, 100), (100, 300)]

    def test_adds_pre_and_post_roll(self):
        seg = expand_event(Event(150, 160, 155, 1.0), self.scenes, 20, 10, 0)
        assert (seg.start, seg.end) == (130, 170)

    def test_pre_roll_never_crosses_a_camera_cut(self):
        seg = expand_event(Event(105, 120, 110, 1.0), self.scenes, 60, 10, 0)
        assert seg.start == 100

    def test_post_roll_never_crosses_a_camera_cut(self):
        seg = expand_event(Event(50, 95, 60, 1.0), self.scenes, 10, 30, 0)
        assert seg.end == 100

    def test_long_clip_is_capped_and_keeps_pre_roll_and_peak(self):
        seg = expand_event(Event(150, 250, 170, 1.0), self.scenes, 20, 10, 50)
        assert seg.length == 50
        assert seg.start == 130  # full pre-roll kept
        assert seg.start <= seg.peak_frame < seg.end

    def test_long_clip_with_late_peak_still_contains_peak(self):
        seg = expand_event(Event(110, 290, 280, 1.0), self.scenes, 10, 5, 50)
        assert seg.length == 50
        assert seg.start <= 280 < seg.end
        assert seg.end <= 300


class TestMerge:
    def test_overlapping_segments_merge_without_duplicate_frames(self):
        merged = merge_segments([Segment(0, 50, 10, 1.0), Segment(40, 90, 60, 3.0)], 0)
        assert len(merged) == 1
        assert (merged[0].start, merged[0].end) == (0, 90)
        assert merged[0].score == 3.0 and merged[0].peak_frame == 60

    def test_small_gap_merges_large_gap_does_not(self):
        segs = [Segment(0, 10, 5, 1), Segment(15, 20, 17, 1), Segment(100, 110, 105, 1)]
        merged = merge_segments(segs, 5)
        assert [(s.start, s.end) for s in merged] == [(0, 20), (100, 110)]

    def test_unsorted_input(self):
        merged = merge_segments([Segment(100, 110, 105, 1), Segment(0, 10, 5, 1)], 0)
        assert [s.start for s in merged] == [0, 100]

    def test_contained_segment(self):
        merged = merge_segments([Segment(0, 100, 5, 1), Segment(10, 20, 15, 2)], 0)
        assert [(s.start, s.end) for s in merged] == [(0, 100)]


class TestBudget:
    def test_keeps_best_that_fit_in_chronological_order(self):
        segs = [
            Segment(0, 40, 10, 1.0),
            Segment(100, 140, 110, 5.0),
            Segment(200, 240, 210, 3.0),
            Segment(300, 330, 310, 2.0),
        ]
        chosen = select_within_budget(segs, 100)
        # 5.0 (40) + 3.0 (40) = 80; 2.0 (30) is trimmed to the last 20; 1.0 doesn't fit
        assert [s.score for s in chosen] == [5.0, 3.0, 2.0]
        assert [s.start for s in chosen] == [100, 200, 300]
        assert sum(s.length for s in chosen) == 100
        assert chosen[2].start <= 310 < chosen[2].end

    def test_skips_one_that_does_not_fit_but_takes_smaller_later(self):
        segs = [Segment(0, 60, 1, 5.0), Segment(100, 200, 101, 4.0), Segment(200, 230, 201, 3.0)]
        chosen = select_within_budget(segs, 100)
        # 4.0 is 100 frames with only 40 left: trimming would cut more than half, so skip.
        assert [s.score for s in chosen] == [5.0, 3.0]

    def test_slightly_too_long_top_play_is_trimmed_not_dropped(self):
        segs = [Segment(0, 64, 30, 5.0), Segment(200, 230, 210, 1.0)]
        chosen = select_within_budget(segs, 60)
        assert [s.score for s in chosen] == [5.0]
        assert chosen[0].length == 60 and chosen[0].start <= 30 < chosen[0].end

    def test_no_budget_keeps_everything(self):
        segs = [Segment(50, 60, 55, 1), Segment(0, 10, 5, 2)]
        assert [s.start for s in select_within_budget(segs, None)] == [0, 50]

    def test_single_huge_segment_is_trimmed_around_its_peak(self):
        chosen = select_within_budget([Segment(0, 1000, 700, 1.0)], 100)
        assert len(chosen) == 1
        assert chosen[0].length == 100
        assert chosen[0].start <= 700 < chosen[0].end


class TestPlanHighlights:
    def test_end_to_end_on_synthetic_curve(self):
        n = 600  # 60 s at 10 fps
        scores = np.full(n, 0.01)
        scores[100:130] = 0.6  # big play in scene 0
        scores[400:420] = 0.3  # smaller play in scene 1
        scenes = [(0, 300), (300, 600)]
        config = Config(
            smoothing_seconds=0.3, percentile=90, min_threshold=0.05,
            pre_roll_seconds=2, post_roll_seconds=1, max_duration_seconds=None,
        )
        segments, threshold, _ = plan_highlights(scores, scenes, FPS, config)
        assert threshold >= 0.05
        assert len(segments) == 2
        first, second = segments
        assert first.start <= 100 - 15 and first.end >= 130
        assert first.score > second.score
        assert second.start >= 300  # clamped to its own scene

    def test_budget_keeps_only_the_bigger_play(self):
        scores = np.full(600, 0.01)
        scores[100:130] = 0.6
        scores[400:420] = 0.3
        config = Config(pre_roll_seconds=2, post_roll_seconds=1, max_duration_seconds=6)
        segments, _, _ = plan_highlights(scores, [(0, 600)], FPS, config)
        assert len(segments) == 1
        assert segments[0].start <= 130 and segments[0].end >= 100

    def test_flat_game_yields_nothing(self):
        segments, _, _ = plan_highlights(np.full(300, 0.01), [(0, 300)], FPS, Config())
        assert segments == []
