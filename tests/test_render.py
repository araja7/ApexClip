import shutil

import pytest

from apexclip.config import Config
from apexclip.renderer import RenderError, has_audio, probe, render_highlights
from apexclip.selector import Segment
from synthetic import make_testsrc

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

NTSC = 30000 / 1001


def stream_codecs(path):
    return {s["codec_type"]: s["codec_name"] for s in probe(path)["streams"]}


def duration(path):
    return float(probe(path)["format"]["duration"])


def test_renders_segments_with_audio_at_fractional_fps(tmp_path):
    src = str(tmp_path / "game.mp4")
    make_testsrc(src, seconds=10, rate="30000/1001")
    out = str(tmp_path / "out" / "reel.mp4")
    segments = [Segment(30, 90, 60, 1.0), Segment(150, 240, 200, 2.0)]

    render_highlights(src, segments, NTSC, out, Config(), log=lambda *a: None)

    assert stream_codecs(out) == {"video": "h264", "audio": "aac"}
    expected = (60 + 90) / NTSC  # ~5.005s
    assert duration(out) == pytest.approx(expected, abs=0.15)


def test_renders_video_without_audio_track(tmp_path):
    src = str(tmp_path / "silent.mp4")
    make_testsrc(src, seconds=4, rate="25", with_audio=False)
    assert not has_audio(src)
    out = str(tmp_path / "reel.mp4")

    render_highlights(src, [Segment(25, 75, 50, 1.0)], 25.0, out, Config(), log=lambda *a: None)

    assert stream_codecs(out) == {"video": "h264"}
    assert duration(out) == pytest.approx(2.0, abs=0.1)


def test_no_segments_is_an_error(tmp_path):
    with pytest.raises(RenderError):
        render_highlights("unused.mp4", [], 30.0, str(tmp_path / "x.mp4"), Config())
