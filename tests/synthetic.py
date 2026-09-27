"""Build small synthetic "games" for integration tests."""
import os
import subprocess

import cv2
import numpy as np

W, H = 640, 360


def _field(width, height, seed, tint):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 90, (height // 6, width // 6, 1), dtype=np.uint8)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_NEAREST)[..., None]
    field = np.clip(noise + np.array(tint, dtype=np.int16), 0, 255).astype(np.uint8)
    # yard lines give the flow estimator strong background features
    for x in range(0, width, 80):
        cv2.line(field, (x, 0), (x, height), (235, 235, 235), 2)
    return field


def _player_sprite():
    import ultralytics

    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    crop = bus[398:902, 48:245]  # a pedestrian YOLO detects with high confidence
    return cv2.resize(crop, (74, 190))


def _paste(canvas, sprite, x, y):
    h, w = sprite.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = sprite[y0 - y:y1 - y, x0 - x:x1 - x]


def make_game(path, fps=15, seconds_per_part=4, with_audio=True):
    """Write a video with four parts and return their frame ranges:

    idle:  camera still, players standing              -> not a highlight
    pan:   camera pans fast, players standing on field -> not a highlight (camera motion)
    play:  camera still, players sprinting             -> THE highlight
    cut:   hard cut to a different field, players idle -> not a highlight
    """
    n = int(fps * seconds_per_part)
    sprite = _player_sprite()
    wide = _field(W * 3, H, seed=1, tint=(30, 110, 30))
    other = _field(W, H, seed=2, tint=(40, 60, 120))
    # Player x positions are in field coordinates; the camera window slides over the field.
    players = [[150.0, 120.0], [330.0, 150.0], [520.0, 110.0], [720.0, 140.0], [880.0, 120.0]]
    pan_speed = 6  # px/frame -> the camera moves 360px over the pan part
    run_speed = 9  # px/frame, reversing direction every second so players stay in view

    frames = []
    cam_x = 0.0
    for part in ("idle", "pan", "play", "cut"):
        for i in range(n):
            if part == "pan":
                cam_x += pan_speed
            if part == "play":
                direction = 1 if (i // fps) % 2 == 0 else -1
                for k, p in enumerate(players):
                    p[0] += run_speed * direction * (1 if k % 2 == 0 else -1)
            if part == "cut":
                frame = other.copy()
                offset = 0
            else:
                frame = wide[:, int(cam_x):int(cam_x) + W].copy()
                offset = int(cam_x)
            for x, y in players:
                _paste(frame, sprite, x - offset, y)
            frames.append(frame)

    raw = path + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", raw]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={len(frames) / fps}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", path]
    subprocess.run(cmd, check=True)
    os.remove(raw)

    return {
        "fps": fps,
        "total": len(frames),
        "idle": (0, n),
        "pan": (n, 2 * n),
        "play": (2 * n, 3 * n),
        "cut": (3 * n, 4 * n),
    }


def make_testsrc(path, seconds=10, rate="30000/1001", with_audio=True):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
           "-i", f"testsrc2=size=320x240:rate={rate}:duration={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", path]
    subprocess.run(cmd, check=True)
