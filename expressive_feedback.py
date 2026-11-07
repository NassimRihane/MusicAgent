# expressive_feedback.py
#
# Layer 4 — Expressive feedback (the research contribution)
#
# Everything before this layer is standard (recognition.py, expression_analysis.py,
# emotion_model.py). This module holds the project's own design choices, kept
# together so they can be discussed, evaluated and changed:
#
#   1. What the listener hears   the emotion Random Forest's prediction for the
#                                performance (emotion_model.predict).
#   2. Where the gap is          for each cue the learner controls, the distance
#                                between their value and the typical value of the
#                                target emotion in EMOPIA (median), in units of
#                                that emotion's inter-quartile range.
#   3. What to work on first     gap size × the model's permutation importance
#                                for that cue; the top MAX_ADVICE cues with a
#                                gap above MIN_GAP become advice.
#   4. Would it help?            a "what-if": the model is re-run with those cues
#                                moved to the target's typical value, and the
#                                change in the target emotion's probability is
#                                reported.
#
# Cues fixed by the piece (mode, register, texture) are reported but never
# turned into advice: a learner can't change the key of the piece they play.
#
# Output: expressive_<session>.json (read by agent.py)
#
# Usage:
#   python expressive_feedback.py sessions/session_X.json --emotion sad
#   python expressive_feedback.py take.mp3 --emotion happy


import argparse
import json
from pathlib import Path

import emotion_model as em
import expression_analysis as ea



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Expressive feedback toward a target emotion")
    parser.add_argument("input", help="Session JSON or audio file")
    parser.add_argument("--emotion", required=True,
                        help=f"Intended emotion: {', '.join(sorted(em.EMOTION_TO_QUADRANT))}")
    parser.add_argument("--out", default=None, help="Output JSON path")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. DESIGN CHOICES
# ─────────────────────────────────────────────

# Cues the performer controls, and how to say "more" / "less" of each
ACTIONABLE = {
    "tempo_bpm":          ("Tempo",             "Play faster",
                                                "Play slower"),
    "velocity_mean":      ("Loudness",          "Play louder, with more weight",
                                                "Play softer, with a lighter touch"),
    "velocity_std":       ("Dynamic contrast",  "Use more contrast between soft and loud notes",
                                                "Keep the dynamics more even"),
    "articulation":       ("Articulation",      "Connect the notes more (legato, or more pedal)",
                                                "Detach the notes more (shorter, lighter touch)"),
    "timing_variability": ("Timing freedom",    "Take more freedom with the timing (rubato)",
                                                "Keep a steadier pulse"),
}

# Cues set by the piece itself: reported, never given as advice
FIXED_BY_PIECE = {
    "major_minor": "Mode (major / minor)",
    "mean_pitch":  "Register",
    "pitch_range": "Pitch range",
    "polyphony":   "Number of simultaneous notes",
    "onset_rate":  "Note density",
}

MIN_GAP    = 0.5   # gaps below half an inter-quartile range are "already there"
MAX_ADVICE = 3



# ─────────────────────────────────────────────
# 2. ANALYSIS
# ─────────────────────────────────────────────

def resolve_target(emotion: str) -> str:
    """'sad' → 'Q3'; also accepts 'Q1'..'Q4'."""
    e = emotion.strip().lower()
    if e.upper() in em.QUADRANTS:
        return e.upper()
    if e not in em.EMOTION_TO_QUADRANT:
        raise ValueError(f"Unknown emotion '{emotion}'. Use one of: "
                         f"{', '.join(sorted(em.EMOTION_TO_QUADRANT))}")
    return em.EMOTION_TO_QUADRANT[e]


def gap(value: float, profile: dict) -> float:
    """Signed distance to the target median, in inter-quartile ranges."""
    iqr = max(profile["q75"] - profile["q25"], 1e-6)
    return (value - profile["median"]) / iqr


def analyse(features: dict, target: str, bundle: dict) -> dict:
    """Prediction, gaps, prioritised advice and what-if for one performance."""
    probs = em.predict(features, bundle)
    profile = bundle["profiles"][target]
    importance = bundle["metrics"]["importance"]

    cues = []
    for cue, (name, more, less) in ACTIONABLE.items():
        g = gap(features[cue], profile[cue])
        cues.append({
            "cue": cue, "name": name,
            "value": round(features[cue], 2), "target": profile[cue]["median"],
            "gap": round(g, 2), "importance": importance.get(cue, 0.0),
            "priority": round(abs(g) * max(importance.get(cue, 0.0), 0.0), 4),
            "advice": less if g > 0 else more,
        })
    cues.sort(key=lambda c: -c["priority"])
    advice = [c for c in cues if abs(c["gap"]) >= MIN_GAP][:MAX_ADVICE]

    what_if = dict(features)
    for c in advice:
        what_if[c["cue"]] = profile[c["cue"]]["median"]
    after = em.predict(what_if, bundle)

    fixed = [{"cue": cue, "name": name, "value": round(features[cue], 2),
              "target": profile[cue]["median"],
              "gap": round(gap(features[cue], profile[cue]), 2)}
             for cue, name in FIXED_BY_PIECE.items()]

    heard = max(probs, key=probs.get)
    return {
        "target":           target,
        "target_name":      em.QUADRANTS[target],
        "heard_as":         heard,
        "heard_as_name":    em.QUADRANTS[heard],
        "probabilities":    probs,
        "advice":           advice,
        "already_there":    [c for c in cues if abs(c["gap"]) < MIN_GAP],
        "what_if": {
            "cues_changed":       [c["cue"] for c in advice],
            "target_probability": {"before": probs[target], "after": after[target]},
        },
        "fixed_by_piece":   fixed,
        "features":         features,
        "model":            bundle["trained_on"],
    }



# ─────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    target = resolve_target(args.emotion)
    notes, _ = ea.load_input(args.input)
    features = ea.note_features(notes)
    if any(v is None for v in features.values()):
        print("ERROR: not enough notes to analyse expression.")
        raise SystemExit(1)

    result = {"input": Path(args.input).name, "emotion": args.emotion,
              **analyse(features, target, em.load())}
    stem = Path(args.input).stem
    out = Path(args.out) if args.out else Path(args.input).with_name(f"expressive_{stem}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    p = result["probabilities"]
    print(f"\n  Intention : {args.emotion} ({target}, {result['target_name']})")
    print(f"  Heard as  : {result['heard_as']} {result['heard_as_name']} "
          f"({p[result['heard_as']]:.0%}); target {target} at {p[target]:.0%}")
    for c in result["advice"]:
        print(f"  → {c['advice']}  [{c['name']}: {c['value']} vs typical {c['target']}]")
    w = result["what_if"]["target_probability"]
    if result["advice"]:
        print(f"  What-if   : {target} probability {w['before']:.0%} → {w['after']:.0%}")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
