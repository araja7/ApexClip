# ApexClip
Turns sports games into a short highlight video.

Give it a full game recording and it finds the moments where players suddenly burst into motion (a snap, a fast break, a breakaway run), cuts those plays out of the original footage with their audio, and stitches the best ones into a reel of the length you choose.

## Quick start

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg` on macOS).

```bash
pip install -r requirements.txt          # or: pip install -e ".[dev]"
python -m apexclip data/raw/game.mp4 -o data/output/game_highlights.mp4 --max-duration 90
```

If you install the package (`pip install -e .`) you can also run `apexclip ...` directly. The original script entry point still works: `python src/main.py` runs on `data/raw/seahawks_rams_game.mp4`.

On Apple Silicon, add `--device mps` for GPU inference, and `--stride 2` to analyze every other frame (roughly 2x faster, usually just as accurate).

## How it works

```
 game.mp4 ─► 1. Analyze ────────────────► 2. Select ──────────────► 3. Render ─► highlights.mp4
             camera cuts (PySceneDetect)     smooth the score          ffmpeg cuts each clip
             players + ball (YOLOv8)         find plays (hysteresis)   from the original, with
             tracks (ByteTrack)              add pre/post-roll         audio, H.264/AAC, then
             remove camera motion            merge overlaps            joins them
             => excitement score per frame   fit the length budget
                   │
                   └─► cached to <output>.analysis.json
```

**1. Analyze** (`src/apexclip/analyzer.py`): the slow pass, done once per video.
- Every camera cut is detected up front, and tracking state is wiped at each cut so a player's position in one shot is never compared with a different shot.
- People and the ball are tracked with YOLOv8 + ByteTrack.
- **Camera motion is removed.** Broadcast cameras pan and zoom constantly, which makes every player look like they're moving. For each frame ApexClip estimates the camera's own movement from background features (optical flow with the players masked out) and subtracts it, so only real on-field motion counts.
- Speeds are measured in *frame-diagonals per second*, so the same thresholds work for 720p, 1080p or 4K at any frame rate.
- A frame's **excitement score** is the average speed of its 5 fastest tracks. Using only the fastest tracks means idle players, referees and fans in the stands don't water down a play where a few players are sprinting.

**2. Select** (`src/apexclip/selector.py`): instant, and runs on the cached scores.
- The score is smoothed, then plays are found with a two-level threshold. A play starts when motion goes above the *threshold*, which by default is the game's own 90th percentile, and continues until it drops below 60% of that. This keeps one long play in one piece.
- Each play gets pre-roll (the setup) and post-roll (the reaction), clamped so a clip never crosses a camera cut.
- Overlapping clips are merged so no frame appears twice. The highest-scoring plays are kept until the `--max-duration` budget is full, then put back in chronological order.

**3. Render** (`src/apexclip/renderer.py`): ffmpeg cuts every clip from the *original* file, so there are no tracking boxes and the audio is kept, and joins them into a browser/phone-friendly H.264/AAC mp4.

## Tuning

The analysis is cached, so after the first run you can re-tune selection in seconds. Use `--dry-run` to see which clips would be picked without rendering.

| Want | Try |
|---|---|
| A longer or shorter reel | `--max-duration 180` (seconds, `0` = keep every play) |
| More, smaller plays | `--percentile 80` |
| Only the very biggest plays | `--percentile 97`, or a fixed `--threshold 0.2` |
| Plays are cut into pieces | `--release-ratio 0.4` or `--smoothing 1.0` |
| More lead-up before a play | `--pre-roll 8` |
| Faster analysis | `--stride 2` or `--stride 3`, `--device mps` / `--device cuda:0` |
| See what the model sees | `--diagnostic data/output/diagnostics/debug.mp4` (boxes, IDs, speeds, score bar) |

Changing `--model`, `--conf` or `--stride` re-runs analysis automatically. Use `--reanalyze` to force it.

## Project layout

```
src/apexclip/
  cli.py        command-line interface
  config.py     every tunable setting with its default
  scenes.py     camera-cut detection
  analyzer.py   YOLO tracking, camera-motion compensation, per-frame scores, cache
  selector.py   scores -> highlight segments (pure functions)
  renderer.py   ffmpeg cutting and joining
src/main.py     original entry point (runs the sample game)
tests/          pytest suite
```

## Tests

```bash
python -m pytest              # everything (~10 s)
python -m pytest -m "not slow"  # skip the tests that run YOLO
```

The end-to-end test builds a synthetic game where players stand still, then the camera pans quickly, then players sprint, then there's a hard cut. It checks that the pan is ignored, the cut is found, and only the sprint ends up in the reel.

## Limitations

- "Exciting" means "lots of fast player movement". Slow but important moments (a free throw, a penalty kick run-up) score low, and replays shown in the broadcast can be picked up as plays.
- Detection uses the stock COCO `yolov8n.pt` model. A larger model (`--model yolov8s.pt`) is more accurate on wide shots with small players, but slower.
