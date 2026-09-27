import cv2
import numpy as np
import pytest

from apexclip.analyzer import (
    Analysis,
    compensated_displacement,
    estimate_camera_motion,
    frame_score,
    interpolate_scores,
    load_cached,
    scale_transform,
)
from apexclip.config import Config
from apexclip.scenes import normalize_scenes, scene_index_for_frame


def textured_image(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)


def shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


class TestCameraMotion:
    def test_recovers_camera_pan(self):
        a = textured_image()
        b = shift(a, 6, -4)
        m = estimate_camera_motion(a, b)
        assert m[0, 2] == pytest.approx(6, abs=0.5)
        assert m[1, 2] == pytest.approx(-4, abs=0.5)
        assert m[0, 0] == pytest.approx(1, abs=0.02)

    def test_static_camera_is_identity(self):
        a = textured_image()
        m = estimate_camera_motion(a, a.copy())
        assert np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=0.05)

    def test_blank_frame_falls_back_to_identity(self):
        blank = np.zeros((120, 160), dtype=np.uint8)
        assert np.array_equal(estimate_camera_motion(blank, blank), [[1, 0, 0], [0, 1, 0]])

    def test_mask_excludes_moving_object(self):
        # Background pans by +5px; a large patch moves by +40px. Masking the patch
        # should leave the estimate on the background motion.
        a = textured_image(seed=1)
        b = shift(a, 5, 0)
        patch = textured_image(80, 80, seed=2)
        a[80:160, 40:120] = patch
        b[80:160, 80:160] = patch
        mask = np.full(a.shape, 255, np.uint8)
        mask[70:170, 30:130] = 0
        m = estimate_camera_motion(a, b, mask)
        assert m[0, 2] == pytest.approx(5, abs=0.7)

    def test_scale_transform_back_to_full_resolution(self):
        m = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]])
        full = scale_transform(m, 0.5)
        assert full[0, 2] == 6.0 and full[1, 2] == -4.0
        assert m[0, 2] == 3.0  # input not mutated


class TestDisplacement:
    identity = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    def test_static_camera(self):
        assert compensated_displacement((0, 0), (3, 4), self.identity) == pytest.approx(5)

    def test_camera_pan_cancels_out(self):
        pan = np.array([[1.0, 0, 20], [0, 1.0, -5]])
        # Player standing still on the field: appears to move exactly with the pan.
        assert compensated_displacement((100, 100), (120, 95), pan) == pytest.approx(0)

    def test_player_running_against_pan(self):
        pan = np.array([[1.0, 0, 20], [0, 1.0, 0]])
        assert compensated_displacement((100, 100), (100, 100), pan) == pytest.approx(20)


class TestScoring:
    def test_frame_score_uses_fastest_tracks(self):
        assert frame_score([0.0, 0.0, 0.0, 1.0, 3.0], top_k=2) == pytest.approx(2.0)

    def test_frame_score_empty(self):
        assert frame_score([], 5) == 0.0

    def test_interpolation_fills_strided_frames(self):
        out = interpolate_scores([0, 2, 4], [0.0, 2.0, 4.0], 6)
        assert np.allclose(out, [0, 1, 2, 3, 4, 4])

    def test_interpolation_without_samples(self):
        assert np.array_equal(interpolate_scores([], [], 3), np.zeros(3))


class TestScenes:
    def test_normalize_fills_gaps_and_covers_video(self):
        assert normalize_scenes([(10, 50), (50, 80)], 100) == [(0, 10), (10, 50), (50, 100)]

    def test_no_cuts_is_one_scene(self):
        assert normalize_scenes([], 42) == [(0, 42)]

    def test_scene_lookup(self):
        scenes = [(0, 10), (10, 20)]
        assert scene_index_for_frame(scenes, 0) == 0
        assert scene_index_for_frame(scenes, 10) == 1
        assert scene_index_for_frame(scenes, 99) == 1


class TestCache:
    def _analysis(self, video, config):
        from apexclip.analyzer import cache_settings, video_signature
        return Analysis(
            video_path=str(video), fps=29.97, total_frames=3, width=10, height=10,
            scenes=[(0, 3)], scores=np.array([0.1, 0.2, 0.3]),
            settings=cache_settings(config), video_signature=video_signature(str(video)),
        )

    def test_round_trip_and_invalidation(self, tmp_path):
        video = tmp_path / "game.mp4"
        video.write_bytes(b"not really a video")
        cache = tmp_path / "game.analysis.json"
        config = Config()
        self._analysis(video, config).save(str(cache))

        loaded = load_cached(str(cache), str(video), config)
        assert loaded is not None
        assert loaded.fps == 29.97 and loaded.scenes == [(0, 3)]
        assert np.allclose(loaded.scores, [0.1, 0.2, 0.3])

        # Selection-only settings don't invalidate the cache...
        assert load_cached(str(cache), str(video), Config(percentile=50, pre_roll_seconds=1)) is not None
        # ...but analysis settings do.
        assert load_cached(str(cache), str(video), Config(stride=3)) is None
        # So does a different video.
        video.write_bytes(b"a different, longer file")
        assert load_cached(str(cache), str(video), config) is None

    def test_corrupt_cache_is_ignored(self, tmp_path):
        video = tmp_path / "game.mp4"
        video.write_bytes(b"x")
        cache = tmp_path / "c.json"
        cache.write_text("{not json")
        assert load_cached(str(cache), str(video), Config()) is None
