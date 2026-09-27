import argparse
import os
import sys
from typing import List, Optional

from .config import Config


def format_time(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        prog="apexclip",
        description="Turn a full sports game video into a short highlight reel.",
    )
    p.add_argument("input", help="path to the game video")
    p.add_argument("-o", "--output", help="highlight reel path (default: <input>_highlights.mp4)")

    g = p.add_argument_group("reel")
    g.add_argument("--max-duration", type=float, default=d.max_duration_seconds,
                   help="target maximum reel length in seconds; 0 for no limit (default: %(default)s)")
    g.add_argument("--pre-roll", type=float, default=d.pre_roll_seconds,
                   help="seconds kept before motion starts (default: %(default)s)")
    g.add_argument("--post-roll", type=float, default=d.post_roll_seconds,
                   help="seconds kept after motion ends (default: %(default)s)")
    g.add_argument("--max-clip", type=float, default=d.max_clip_seconds,
                   help="maximum length of a single clip in seconds (default: %(default)s)")

    g = p.add_argument_group("detection sensitivity")
    g.add_argument("--threshold", type=float, default=None,
                   help="absolute excitement threshold in frame-diagonals/second "
                        "(default: adaptive, see --percentile)")
    g.add_argument("--percentile", type=float, default=d.percentile,
                   help="adaptive threshold: frames above this percentile of the game's own "
                        "motion count as action (default: %(default)s)")
    g.add_argument("--release-ratio", type=float, default=d.release_ratio,
                   help="a play continues while motion stays above threshold x this ratio "
                        "(default: %(default)s)")
    g.add_argument("--min-event", type=float, default=d.min_event_seconds,
                   help="seconds motion must stay above threshold (default: %(default)s)")
    g.add_argument("--smoothing", type=float, default=d.smoothing_seconds,
                   help="moving-average window in seconds (default: %(default)s)")

    g = p.add_argument_group("analysis")
    g.add_argument("--model", default=d.model_path, help="YOLO weights (default: %(default)s)")
    g.add_argument("--conf", type=float, default=d.detection_conf,
                   help="detection confidence (default: %(default)s)")
    g.add_argument("--device", default=None, help="inference device, e.g. mps, cuda:0, cpu")
    g.add_argument("--stride", type=int, default=d.stride,
                   help="analyze every Nth frame; 2-3 is much faster (default: %(default)s)")
    g.add_argument("--diagnostic", metavar="PATH",
                   help="also write an annotated debug video with boxes and scores")
    g.add_argument("--cache", metavar="PATH",
                   help="analysis cache file (default: <output>.analysis.json)")
    g.add_argument("--no-cache", action="store_true", help="don't read or write the analysis cache")
    g.add_argument("--reanalyze", action="store_true", help="ignore any existing cache")

    p.add_argument("--dry-run", action="store_true",
                   help="analyze and print the chosen clips without rendering")
    return p


def config_from_args(args) -> Config:
    return Config(
        model_path=args.model,
        detection_conf=args.conf,
        device=args.device,
        stride=max(1, args.stride),
        smoothing_seconds=args.smoothing,
        threshold=args.threshold,
        percentile=args.percentile,
        release_ratio=args.release_ratio,
        min_event_seconds=args.min_event,
        pre_roll_seconds=args.pre_roll,
        post_roll_seconds=args.post_roll,
        max_clip_seconds=args.max_clip,
        max_duration_seconds=args.max_duration if args.max_duration and args.max_duration > 0 else None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    if not os.path.isfile(args.input):
        print(f"Error: input video not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or f"{os.path.splitext(args.input)[0]}_highlights.mp4"
    cache_path = None if args.no_cache else (args.cache or f"{os.path.splitext(output)[0]}.analysis.json")

    # Imported here so `--help` stays fast.
    from .analyzer import analyze, load_cached
    from .renderer import RenderError, render_highlights, require_ffmpeg
    from .selector import plan_highlights

    if not args.dry_run:
        try:
            require_ffmpeg()  # fail before the long analysis pass, not after
        except RenderError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    print("--- ApexClip ---")
    analysis = None
    if cache_path and not args.reanalyze and not args.diagnostic:
        analysis = load_cached(cache_path, args.input, config)
        if analysis:
            print(f"Using cached analysis: {cache_path}")
    if analysis is None:
        analysis = analyze(args.input, config, diagnostic_path=args.diagnostic)
        if cache_path:
            analysis.save(cache_path)
            print(f"Saved analysis cache: {cache_path}")
        if args.diagnostic:
            print(f"Diagnostic video: {args.diagnostic}")

    segments, threshold, smoothed = plan_highlights(
        analysis.scores, analysis.scenes, analysis.fps, config
    )
    fps = analysis.fps
    game_seconds = analysis.total_frames / fps
    print(f"\nGame length: {format_time(game_seconds)} | {len(analysis.scenes)} scenes | "
          f"threshold {threshold:.3f} (peak motion {smoothed.max() if len(smoothed) else 0:.3f})")

    if not segments:
        print("No highlights found. Try a lower --percentile or --threshold.")
        return 2

    total = sum(s.length for s in segments) / fps
    print(f"Selected {len(segments)} clip(s), {format_time(total)} total:")
    for n, s in enumerate(segments, 1):
        print(f"  {n:3d}. {format_time(s.start / fps)} - {format_time(s.end / fps)}  "
              f"({s.length / fps:5.1f}s)  score {s.score:.3f}")

    if args.dry_run:
        return 0

    print("\nRendering highlight reel...")
    try:
        render_highlights(args.input, segments, fps, output, config)
    except RenderError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"\nHighlight reel saved to {output}")
    return 0
