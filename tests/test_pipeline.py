"""End-to-end: real YOLO + tracking + scene detection + selection + ffmpeg render
on a synthetic game where players stand, the camera pans, players sprint, then a cut."""
import os
import shutil

import numpy as np
import pytest

from apexclip.analyzer import Analysis
from apexclip.cli import main
from apexclip.config import Config
from apexclip.renderer import probe
from apexclip.selector import plan_highlights
from synthetic import make_game

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "yolov8n.pt")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(MODEL), reason="yolov8n.pt not present"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    video = str(tmp / "game.mp4")
    info = make_game(video)
    output = str(tmp / "reel.mp4")
    diagnostic = str(tmp / "diag.mp4")
    code = main([
        video, "-o", output, "--model", MODEL, "--pre-roll", "1", "--post-roll", "1",
        "--max-duration", "0", "--diagnostic", diagnostic,
    ])
    return {
        "code": code, "info": info, "video": video, "output": output, "diagnostic": diagnostic,
        "analysis": Analysis.load(os.path.splitext(output)[0] + ".analysis.json"),
    }


def part_scores(run, part):
    start, end = run["info"][part]
    return run["analysis"].scores[start:end]


def test_cli_succeeds_and_writes_outputs(run):
    assert run["code"] == 0
    assert os.path.exists(run["output"])
    assert os.path.getsize(run["diagnostic"]) > 0
    codecs = {s["codec_type"]: s["codec_name"] for s in probe(run["output"])["streams"]}
    assert codecs == {"video": "h264", "audio": "aac"}


def test_hard_cut_is_detected(run):
    cut = run["info"]["cut"][0]
    starts = [s for s, _ in run["analysis"].scenes]
    assert any(abs(s - cut) <= 2 for s in starts), starts


def test_camera_pan_is_not_mistaken_for_action(run):
    pan = part_scores(run, "pan").mean()
    play = part_scores(run, "play").mean()
    idle = part_scores(run, "idle").mean()
    # Without compensation the pan would score ~0.12 diagonals/s, close to the play.
    assert pan < 0.06
    assert play > 3 * pan
    assert play > 10 * idle


def test_selected_highlight_is_the_play(run):
    a = run["analysis"]
    config = Config(pre_roll_seconds=1, post_roll_seconds=1, max_duration_seconds=None)
    segments, _, _ = plan_highlights(a.scores, a.scenes, a.fps, config)
    play_start, play_end = run["info"]["play"]
    assert segments
    for seg in segments:
        assert play_start <= seg.peak_frame < play_end
        assert seg.end <= play_end  # never runs past the camera cut into the next shot
    reel = float(probe(run["output"])["format"]["duration"])
    assert reel == pytest.approx(sum(s.length for s in segments) / a.fps, abs=0.2)


def test_second_run_uses_cache(run, capsys):
    code = main([run["video"], "-o", run["output"], "--model", MODEL, "--dry-run",
                 "--pre-roll", "1", "--post-roll", "1", "--percentile", "80"])
    assert code == 0
    assert "Using cached analysis" in capsys.readouterr().out


def test_flat_video_reports_no_highlights(tmp_path, capsys):
    from synthetic import make_testsrc
    # testsrc has no people, so there's nothing to track
    video = str(tmp_path / "empty.mp4")
    make_testsrc(video, seconds=2, rate="15")
    code = main([video, "--model", MODEL, "--dry-run", "--no-cache"])
    assert code == 2
    assert "No highlights found" in capsys.readouterr().out
