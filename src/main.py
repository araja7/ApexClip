"""Backwards-compatible entry point: runs ApexClip on the original sample paths.

Prefer the CLI for anything else:  python -m apexclip <video> -o <output>
"""
import sys

from apexclip.cli import main

if __name__ == "__main__":
    sys.exit(main([
        "data/raw/seahawks_rams_game.mp4",
        "-o", "data/output/seahawks_rams_highlights.mp4",
        "--diagnostic", "data/output/diagnostics/ai_annotated_seahawks_rams_game.mp4",
        *sys.argv[1:],
    ]))
