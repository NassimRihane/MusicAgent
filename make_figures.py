# make_figures.py
#
# Figures for presentations and the paper, regenerated from the saved reports
# (nothing is recomputed). Each figure is written as PNG (slides) and PDF
# (vector, for LaTeX) in figures/, with a caption in figures/captions.md.
#
#   emotion_model_comparison   Random Forest vs standard classifiers (EMOPIA)
#   emotion_confusion          confusion matrix of the Random Forest
#   emotion_feature_importance permutation importance of the 10 descriptors
#   chord_eval_comparison      both chord recognisers, rendered progressions
#                              vs a real piano recording
#   chord_eval_by_quality      real recording: share correct per chord type
#   advice_example             what the listener model hears, and the gaps
#                              turned into advice, for one performance
#   piano_roll_demo            piano rolls of two performances
#
# Inputs (missing ones are skipped with a message):
#   models/emotion_rf_report.json          python emotion_model.py --train
#   models/chord_eval_report.json          python chord_evaluation.py
#   models/chord_eval_recording_*.json     python chord_evaluation.py --recording TAKE
#   expressive_*.json                      python expressive_feedback.py ...
#   expression JSON with 2 recordings      python expression_analysis.py A --compare B
#
# Usage:
#   python make_figures.py
#   python make_figures.py --expressive sessions/expressive_session_X.json \
#                          --recording-report models/chord_eval_recording_take.json


import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import emotion_model as em



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

ROOT    = Path(__file__).parent
MODELS  = ROOT / "models"
FIGURES = ROOT / "figures"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Figures for presentations and the paper")
    parser.add_argument("--recording-report", default=None,
                        help="Real-recording chord report (default: the newest in models/)")
    parser.add_argument("--expressive", default=None,
                        help="expressive_*.json for the advice figure "
                             "(default: the newest in sessions/)")
    parser.add_argument("--expression", default=str(ROOT / "sessions" / "demo_happy_birthday"
                                                    / "expression.json"),
                        help="expression JSON with two recordings, for the piano rolls")
    parser.add_argument("--out", default=str(FIGURES), help="Output folder (default figures/)")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. STYLE
# ─────────────────────────────────────────────

INK       = "#14171c"
INK_MUTED = "#6b717b"
GRID      = "#e3e6ea"
NEUTRAL   = "#b8bdc6"
MUSIC21   = "#2a78d6"   # categorical slot 1
RF        = "#eb6834"   # categorical slot 2

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42,   # editable text in the PDF
})

CAPTIONS = {}


def save(fig, name: str, out: Path, caption: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CAPTIONS[name] = caption
    print(f"  {name}")


def percent_axis(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def bar_labels(ax, bars, values, fmt="{:.0%}", inside=False, horizontal=False):
    for bar, v in zip(bars, values):
        if horizontal:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, fmt.format(v),
                    va="center", ha="left", fontsize=9, color=INK)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, fmt.format(v),
                    va="bottom", ha="center", fontsize=7.5, color=INK)



# ─────────────────────────────────────────────
# 2. EMOTION MODEL
# ─────────────────────────────────────────────

def emotion_figures(report: dict, out: Path):
    m = report["metrics"]
    rows = m["comparison"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = [r["model"] for r in rows]
    means = [r["accuracy_mean"] for r in rows]
    stds = [r["accuracy_std"] for r in rows]
    colors = [MUSIC21 if r["model"] == "Random Forest" else NEUTRAL for r in rows]
    y = np.arange(len(rows))[::-1]
    bars = ax.barh(y, means, xerr=stds, color=colors, height=0.6,
                   error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1%} ± {std:.1%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Accuracy, 4 emotion quadrants (mean ± std over 5 song-grouped folds)")
    ax.set_title(f"Emotion recognition — {report['trained_on']}", loc="left")
    rf = rows[-1]
    save(fig, "emotion_model_comparison", out,
         f"Emotion recognition on {report['trained_on']}: accuracy on the four quadrants of "
         "Russell's valence-arousal model, 5-fold cross-validation grouped by song "
         "(no song in both training and test). The Random Forest "
         f"({rf['accuracy_mean']:.1%} ± {rf['accuracy_std']:.1%}) is at least as accurate as the "
         "other standard classifiers; chance level is the majority class.")

    em.save_confusion_figure(m, out / "emotion_confusion.png")
    _confusion_pdf(m, out)   # same figure as PDF for the paper
    CAPTIONS["emotion_confusion"] = (
        "Confusion matrix of the emotion Random Forest (rows: true quadrant, columns: predicted; "
        "row-normalised). Errors are mostly between neighbouring quadrants that share an "
        "arousal level (happy vs tense, sad vs calm).")
    print("  emotion_confusion")

    imp = sorted(m["importance"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.barh([f for f, _ in imp], [max(v, 0) for _, v in imp], color=MUSIC21, height=0.6)
    for bar, (_, v) in zip(bars, imp):
        ax.text(max(v, 0) + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:+.3f}",
                va="center", fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Permutation importance (drop in accuracy when the feature is shuffled)")
    ax.set_title("What the emotion model relies on", loc="left")
    save(fig, "emotion_feature_importance", out,
         "Permutation importance of the ten note-level descriptors, averaged over the test "
         "folds. Loudness (mean MIDI velocity) and mode (major/minor) dominate; tempo and "
         "articulation contribute little.")


def _confusion_pdf(metrics: dict, out: Path):
    """PDF version of emotion_model.save_confusion_figure."""
    cm = np.array(metrics["confusion"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    labels = [f"{c}\n{em.QUADRANTS[c]}" for c in metrics["classes"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.grid(False)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(cm)), labels, fontsize=8)
    ax.set_yticks(range(len(cm)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Random Forest — accuracy {metrics['accuracy']:.1%}")
    fig.savefig(out / "emotion_confusion.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)



# ─────────────────────────────────────────────
# 3. CHORD RECOGNISERS
# ─────────────────────────────────────────────

METRICS = ["root", "majmin", "sevenths"]
METRIC_NAMES = {"root": "Root", "majmin": "Major/minor", "sevenths": "Sevenths"}
BACKENDS = [("music21", "basic-pitch + music21 (notes)", MUSIC21),
            ("rf", "Random Forest (audio chroma)", RF)]

QUALITY_NAMES = {"maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented",
                 "7": "dominant 7th", "maj7": "major 7th", "min7": "minor 7th"}


def chord_figures(rendered: dict, recording: dict, recording_name: str, out: Path):
    panels = []
    if rendered:
        n = rendered["pieces"]
        panels.append((f"Rendered piano progressions\n(in context, {n} pieces, mean ± std)",
                       {b: [rendered["summary"][b]["all"][m]["mean"] for m in METRICS] for b, _, _ in BACKENDS},
                       {b: [rendered["summary"][b]["all"][m]["std"] for m in METRICS] for b, _, _ in BACKENDS}))
    if recording:
        n = len(recording["chords"])
        panels.append((f"Real piano recording\n(isolated chords, {n} chords)",
                       {b: [recording["scores"][b][m] for m in METRICS] for b, _, _ in BACKENDS},
                       None))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8),
                             sharey=True, squeeze=False)
    x = np.arange(len(METRICS))
    width = 0.36
    for ax, (title, means, stds) in zip(axes[0], panels):
        for k, (b, label, color) in enumerate(BACKENDS):
            offset = (k - 0.5) * width
            bars = ax.bar(x + offset, means[b], width * 0.92, color=color, label=label,
                          yerr=stds[b] if stds else None,
                          error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2})
            bar_labels(ax, bars, means[b])
        ax.set_xticks(x, [METRIC_NAMES[m] for m in METRICS])
        ax.set_ylim(0, 1.12)
        percent_axis(ax)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
    axes[0][0].set_ylabel("MIREX score (mir_eval)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    parts = []
    if rendered:
        r = rendered["summary"]
        parts.append(f"on rendered progressions in context, the Random Forest scores "
                     f"{r['rf']['all']['majmin']['mean']:.0%} (major/minor) against "
                     f"{r['music21']['all']['majmin']['mean']:.0%} for the note-based path")
    if recording:
        s = recording["scores"]
        parts.append(f"on a real piano recording of isolated chords, the note-based path "
                     f"recognises {s['music21']['sevenths']:.0%} of the chords including sevenths, "
                     f"against {s['rf']['sevenths']:.0%} for the Random Forest")
    save(fig, "chord_eval_comparison", out,
         "Chord recognition, MIREX scores computed with mir_eval: " + "; ".join(parts) + ". "
         "The note-based path (basic-pitch + music21) is the default because the real recording "
         "is the most direct evidence and its errors are not systematic.")

    if recording:
        chord_quality_figure(recording, recording_name, out)


def chord_quality_figure(recording: dict, name: str, out: Path):
    groups = {}
    for c in recording["chords"]:
        quality = c["expected"].split(":")[1]
        groups.setdefault(quality, []).append(c)
    order = [q for q in QUALITY_NAMES if q in groups]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(order))
    width = 0.36
    for k, (b, label, color) in enumerate(BACKENDS):
        values = [np.mean([c[b] == c["expected"] for c in groups[q]]) for q in order]
        bars = ax.bar(x + (k - 0.5) * width, values, width * 0.92, color=color, label=label)
        bar_labels(ax, bars, values)
    ax.set_xticks(x, [f"{QUALITY_NAMES[q]}\n(n={len(groups[q])})" for q in order], fontsize=9)
    ax.set_ylim(0, 1.15)
    percent_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Chords recognised exactly")
    ax.set_title(f"Real piano recording, by chord type ({name})", loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    rf_sevenths = [c for q in ("7", "maj7", "min7") for c in groups.get(q, [])]
    rf_hits = sum(c["rf"] == c["expected"] for c in rf_sevenths)
    save(fig, "chord_eval_by_quality", out,
         f"Real piano recording of the 28-chord protocol: share of chords recognised exactly, "
         f"by chord type. The Random Forest recognises {rf_hits} of {len(rf_sevenths)} seventh "
         "chords: on real audio it reduces them to triads, a systematic bias from its synthetic "
         "training data. Augmented triads are ambiguous by nature (C, E and G# augmented share "
         "their notes).")



# ─────────────────────────────────────────────
# 4. EXPRESSIVE ADVICE
# ─────────────────────────────────────────────

def advice_figure(x: dict, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    quadrants = list(em.QUADRANTS)
    probs = [x["probabilities"][q] for q in quadrants]
    colors = [MUSIC21 if q == x["target"] else NEUTRAL for q in quadrants]
    bars = ax1.bar(range(4), probs, color=colors, width=0.6)
    bar_labels(ax1, bars, probs)
    after = x["what_if"]["target_probability"]["after"]
    t = quadrants.index(x["target"])
    if x["advice"]:
        ax1.plot([t - 0.3, t + 0.3], [after, after], color=INK, linewidth=2)
        ax1.text(t, after + 0.012, f"after advice\n{after:.0%}", fontsize=8, color=INK,
                 ha="center", va="bottom")
    names = [em.QUADRANTS[q].replace(" / ", " /\n") for q in quadrants]
    ax1.set_xticks(range(4), [f"{q}\n{n}" for q, n in zip(quadrants, names)], fontsize=8)
    ax1.set_ylim(0, max(max(probs), after) + 0.18)
    percent_axis(ax1)
    ax1.grid(axis="x", visible=False)
    ax1.set_title(f"What the listener model hears\n(intention: {x.get('emotion', x['target_name'])})",
                  fontsize=10)

    cues = sorted(x["advice"] + x["already_there"], key=lambda c: -c["priority"])
    advised = {c["cue"] for c in x["advice"]}
    y = np.arange(len(cues))[::-1]
    ax2.barh(y, [c["gap"] for c in cues], height=0.55,
             color=[MUSIC21 if c["cue"] in advised else NEUTRAL for c in cues])
    ax2.axvline(0, color=INK_MUTED, linewidth=1)
    ax2.set_yticks(y, [c["name"] for c in cues])
    span = max(abs(c["gap"]) for c in cues) + 1.8
    ax2.set_xlim(-span, span)
    for yy, c in zip(y, cues):
        text = c["advice"] if c["cue"] in advised else "already close"
        side = -1 if c["gap"] > 0 else 1
        ax2.text(side * 0.1, yy, text, va="center", ha="right" if side < 0 else "left",
                 fontsize=8, color=INK if c["cue"] in advised else INK_MUTED)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel(f"← lower than typical {x['target_name']} performances      "
                   f"higher than typical →\n(gap in inter-quartile ranges)")
    ax2.set_title("Controllable cues: gaps and advice (blue = advised, by priority)", fontsize=10)
    fig.tight_layout()
    w = x["what_if"]["target_probability"]
    save(fig, "advice_example", out,
         f"Expressive feedback for one performance ({x.get('input', '?')}), intention "
         f"\"{x.get('emotion')}\". Left: emotion probabilities from the listener model "
         f"(heard as {x['heard_as_name']}). Right: for each cue the performer controls, the gap "
         f"to typical {x['target_name']} performances in EMOPIA; the largest gaps weighted by the "
         f"model's importance become advice. Applying it would raise the intended emotion from "
         f"{w['before']:.0%} to {w['after']:.0%}.")



# ─────────────────────────────────────────────
# 5. PIANO ROLLS
# ─────────────────────────────────────────────

def piano_roll_figure(expression_path: Path, out: Path):
    import piano_roll
    recordings = piano_roll.load_recordings(expression_path)
    fig = piano_roll.plot_piano_rolls(recordings, title="What the system heard")
    save(fig, "piano_roll_demo", out,
         "Piano rolls of two performances transcribed by basic-pitch (same time scale and pitch "
         "range; fainter bars are softer notes), with the key detected by music21.")



# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def newest(pattern: str, folder: Path):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None


def write_captions(out: Path):
    lines = ["# Figure captions", "",
             "Generated by `make_figures.py` from the saved reports; numbers update when the "
             "figures are regenerated.", ""]
    for name, caption in CAPTIONS.items():
        lines += [f"## {name}", "", f"![{name}]({name}.png)", "", caption, ""]
    (out / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CAPTIONS.clear()
    print(f"  Writing figures to {out}/")

    emotion = load_json(MODELS / "emotion_rf_report.json")
    if emotion and emotion["metrics"].get("comparison"):
        emotion_figures(emotion, out)
    else:
        print("  (skipped emotion figures: run python emotion_model.py --train)")

    rendered = load_json(MODELS / "chord_eval_report.json")
    rec_path = Path(args.recording_report) if args.recording_report else \
               newest("chord_eval_recording_*.json", MODELS)
    recording = load_json(rec_path)
    if rendered or recording:
        name = rec_path.stem.replace("chord_eval_recording_", "") if recording else ""
        chord_figures(rendered, recording, name, out)
    else:
        print("  (skipped chord figures: run python chord_evaluation.py)")

    exp_path = Path(args.expressive) if args.expressive else \
               newest("expressive_*.json", ROOT / "sessions")
    expressive = load_json(exp_path)
    if expressive:
        advice_figure(expressive, out)
    else:
        print("  (skipped advice figure: run expressive_feedback.py on a session)")

    if Path(args.expression).exists():
        piano_roll_figure(Path(args.expression), out)
    else:
        print("  (skipped piano rolls: no expression JSON)")

    write_captions(out)
    print(f"  Captions: {out / 'captions.md'}")


if __name__ == "__main__":
    main()
