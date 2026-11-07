# piano_roll.py
#
# Draws "what the system heard": one piano roll per recording, stacked on a
# shared time scale and pitch axis so performances can be compared at a glance.
# Each bar is a detected note (height = pitch, length = duration); fainter bars
# are softer notes.
#
# Accepted inputs (can be mixed):
#   audio file (.wav, .mp3, ...)  → transcribed with basic-pitch
#   session JSON                  → output of transcription.py
#   expression JSON               → output of expression_analysis.py (no re-transcription)
#
# Usage:
#   python piano_roll.py "Happy Birthday.mp3" "Unhappy Birthday.mp3" --labels Happy Sad
#   python piano_roll.py sessions/session_20260101_120000.json
#   python piano_roll.py sessions/demo_happy_birthday/expression.json --out roll.png
#   python piano_roll.py take1.wav take2.wav --show


import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

import recognition as rc



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Piano-roll comparison of recordings")
    parser.add_argument("inputs", nargs="+",
                        help="Audio files, session JSONs or expression JSONs")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="One label per recording (default: file names)")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--colors", default=None,
                        help="Comma-separated colours, one per recording (default palette otherwise)")
    parser.add_argument("--out", default=None,
                        help="Output image (.png, .pdf, .svg). Default: piano_roll_<first input>.png")
    parser.add_argument("--show", action="store_true", help="Also open the figure in a window")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

# Fixed colour order (colour-blind-checked categorical palette); never cycled.
PALETTE = ["#eb6834", "#2a78d6", "#1baf7a", "#6250d6", "#e34948", "#eda100"]

INK       = "#14171c"
INK_MUTED = "#7a808a"
GRID      = "#e3e6ea"
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']



# ─────────────────────────────────────────────
# 2. LOADING
# ─────────────────────────────────────────────

def load_recordings(path: str) -> list:
    """
    Return a list of {label, notes, duration} for one input.
    An expression JSON with --compare holds two recordings, hence the list.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "recordings" in data:
            return [{"label": Path(r["file"]).stem, "notes": r["notes"],
                     "duration": r["cues"].get("duration_sec")} for r in data["recordings"]]
        if "events" in data or "notes" in data:
            return [{"label": path.stem, "notes": rc.session_notes(data), "duration": None}]
        raise ValueError(f"{path.name}: not a session or expression JSON")

    import librosa
    return [{"label": path.stem, "notes": rc.transcribe(path),
             "duration": librosa.get_duration(path=str(path))}]



# ─────────────────────────────────────────────
# 3. PLOTTING
# ─────────────────────────────────────────────

def describe(rec: dict) -> str:
    """Short summary line: key · duration · note count."""
    notes = rec["notes"]
    if not notes:
        return "no notes detected"
    key = rc.detect_key(notes)["key"]
    return f"{key}  ·  {rec['duration']:.1f} s  ·  {len(notes)} notes"


def plot_piano_rolls(recordings: list, colors: list = None, title: str = None):
    """Draw one piano roll per recording on shared axes. Returns the figure."""
    colors = colors or PALETTE
    if len(recordings) > len(colors):
        raise ValueError(f"At most {len(colors)} recordings per figure (got {len(recordings)})")

    for rec in recordings:
        if rec.get("duration") is None:
            rec["duration"] = max((n["end"] for n in rec["notes"]), default=1.0)

    all_midi = [n["midi"] for rec in recordings for n in rec["notes"]] or [60]
    lo = (min(all_midi) - 1) // 12 * 12
    hi = (max(all_midi) + 2 + 11) // 12 * 12
    t_max = max(rec["duration"] for rec in recordings)

    fig, axes = plt.subplots(len(recordings), 1, sharex=True, sharey=True,
                             figsize=(11, 3.3 * len(recordings) + 0.6), squeeze=False)
    axes = axes[:, 0]

    for ax, rec, color in zip(axes, recordings, colors):
        notes = rec["notes"]
        if notes:
            rgba = np.array([to_rgba(color, 0.25 + 0.75 * min(n["velocity"], 127) / 127)
                             for n in notes])
            ax.barh([n["midi"] for n in notes],
                    [max(n["end"] - n["start"], 0.02) for n in notes],
                    left=[n["start"] for n in notes],
                    height=0.85, color=rgba, linewidth=0)

        # header: colour key + label in ink, summary in muted ink
        ax.add_patch(Rectangle((0, 1.06), 0.012, 0.09, transform=ax.transAxes,
                               color=color, clip_on=False))
        ax.text(0.02, 1.06, rec["label"], transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=INK, va="bottom")
        ax.text(1.0, 1.06, describe(rec), transform=ax.transAxes,
                fontsize=10, color=INK_MUTED, va="bottom", ha="right")

        ax.set_yticks(range(lo, hi + 1, 12))
        ax.set_yticklabels([f"C{m // 12 - 1}" for m in range(lo, hi + 1, 12)])
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    axes[0].set_ylim(lo, hi)
    axes[0].set_xlim(0, t_max)
    axes[-1].set_xlabel("time (s)", color=INK_MUTED, fontsize=10)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95 if title else 1))
    left = axes[0].get_position().x0   # align title and caption with the panels
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, x=left, ha="left")
    fig.text(left, 0.005, "Each bar is a detected note. Fainter bars are softer notes.",
             fontsize=9, color=INK_MUTED)
    return fig



# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")   # no window needed: works without a display

    recordings = [rec for path in args.inputs for rec in load_recordings(path)]
    if args.labels:
        if len(args.labels) != len(recordings):
            raise SystemExit(f"ERROR: {len(args.labels)} labels for {len(recordings)} recordings")
        for rec, label in zip(recordings, args.labels):
            rec["label"] = label

    colors = args.colors.split(",") if args.colors else None
    fig = plot_piano_rolls(recordings, colors, args.title)

    out = Path(args.out) if args.out else \
          Path(args.inputs[0]).with_name(f"piano_roll_{Path(args.inputs[0]).stem}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
