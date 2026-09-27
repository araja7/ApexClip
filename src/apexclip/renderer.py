import json
import os
import shutil
import subprocess
import tempfile
from typing import List

from .config import Config
from .selector import Segment


class RenderError(RuntimeError):
    pass


def require_ffmpeg():
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RenderError(
            f"{' and '.join(missing)} not found on PATH. Install ffmpeg (e.g. `brew install ffmpeg`)."
        )


def _run(cmd: List[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RenderError(f"Command failed: {' '.join(cmd)}\n{tail}")
    return proc


def probe(path: str) -> dict:
    proc = _run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height",
        "-of", "json", path,
    ])
    return json.loads(proc.stdout)


def has_audio(path: str) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def render_highlights(
    video_path: str, segments: List[Segment], fps: float, output_path: str, config: Config, log=print
):
    """Cut each segment from the original video (with its audio) and join them.

    Each clip is re-encoded with identical settings so the final concat is a lossless
    stream copy, and the output is H.264/AAC with faststart so it plays in browsers
    and on phones.
    """
    require_ffmpeg()
    if not segments:
        raise RenderError("No segments to render.")
    audio = has_audio(video_path)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="apexclip_") as tmp:
        clip_paths = []
        for n, seg in enumerate(segments, 1):
            start = seg.start / fps
            duration = seg.length / fps
            clip_path = os.path.join(tmp, f"clip_{n:04d}.mp4")
            log(f"  cutting clip {n}/{len(segments)}: {start:.2f}s - {start + duration:.2f}s")
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
                "-map", "0:v:0",
            ]
            if audio:
                cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
            cmd += [
                "-c:v", "libx264", "-preset", config.preset, "-crf", str(config.crf),
                "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}",
                "-avoid_negative_ts", "make_zero", clip_path,
            ]
            _run(cmd)
            clip_paths.append(clip_path)

        list_path = os.path.join(tmp, "clips.txt")
        with open(list_path, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")
        _run([
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-movflags", "+faststart", output_path,
        ])
