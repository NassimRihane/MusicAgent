# feedback.py
#
# Step 2 — Correction feedback (context for the coach)
#
# Reads a session JSON (from transcription.py), its WAV if present, and the
# chord analysis JSON (from chord_analysis.py, auto-detected), then reports
# three classic levels. All measurements rely on standard tools (recognition.py):
#
#   LEVEL 1 — Notes    : share of notes in the session's key (music21 scale)
#   LEVEL 2 — Rhythm   : tempo and tempo stability (librosa), hesitations.
#                        Without the score, this can't tell rubato from an
#                        unsteady pulse: it measures stability, not correctness.
#   LEVEL 3 — Harmony  : share of in-key chords and known progressions
#                        (music21 Roman numerals)
#
# These corrections are context; the research focus is the expressive
# feedback (expressive_feedback.py).
#
# Output:
#   feedback_<session>.json    — machine-readable feedback for the agent
#   feedback_<session>.txt     — human-readable summary
#
# Usage:
#   python feedback.py sessions/session_20260101_120000.json
#   python feedback.py sessions/session_20260101_120000.json \
#                      --chords sessions/session_20260101_120000_chords.json


import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import librosa

import recognition as rc



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Correction feedback for a session")
    parser.add_argument("session", help="Path to a session JSON file (from transcription.py)")
    parser.add_argument("--chords", default=None,
                        help="Chord analysis JSON (default: <session>_chords.json if it exists)")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

MIN_ATTACKS = 4   # attacks needed to judge the tempo

# Tempo stability (recognition.track_beats tempo_cv). Calibrated on synthetic
# notes: 0 steady, ~0.02 with 20 ms jitter, ~0.06 with 40 ms, ~0.12 for a
# deliberate accelerando–ritardando. A tempo_cv of UNSTABLE_CV scores 0.
STEADY_CV, MOSTLY_STEADY_CV, UNSTABLE_CV = 0.03, 0.08, 0.15

KNOWN_PROGRESSIONS = {
    "I-IV-V":       ["I", "IV", "V"],
    "I-V-vi-IV":    ["I", "V", "vi", "IV"],
    "ii-V-I":       ["ii", "V", "I"],
    "I-vi-IV-V":    ["I", "vi", "IV", "V"],
    "vi-IV-I-V":    ["vi", "IV", "I", "V"],
    "i-iv-V":       ["i", "iv", "V"],
    "i-VI-III-VII": ["i", "VI", "III", "VII"],
    "i-VII-VI-V":   ["i", "VII", "VI", "V"],
}



# ─────────────────────────────────────────────
# 2. LEVEL 1 — NOTES
# ─────────────────────────────────────────────

def analyse_notes(notes: list, scale_pc: set) -> dict:
    """Share of notes in the key, most frequent out-of-key notes, average loudness."""
    if not notes:
        return {"total_notes": 0, "in_scale_ratio": 0.0, "status": "no_data"}
    pcs = [n["midi"] % 12 for n in notes]
    in_scale = sum(pc in scale_pc for pc in pcs)
    ratio = in_scale / len(notes)
    errors = Counter(rc.NOTE_NAMES[pc] for pc in pcs if pc not in scale_pc)

    if ratio >= 0.95:
        assessment = "Excellent: almost all notes are in the key"
    elif ratio >= 0.8:
        assessment = "Good: few notes outside the key"
    elif ratio >= 0.6:
        assessment = "Fair: quite a few notes outside the key"
    else:
        assessment = "Many notes outside the key"

    return {
        "total_notes":        len(notes),
        "in_scale_count":     in_scale,
        "out_of_scale_count": len(notes) - in_scale,
        "in_scale_ratio":     round(ratio, 3),
        "score":              round(ratio * 100),
        "assessment":         assessment,
        "frequent_errors":    [{"note": n, "count": c} for n, c in errors.most_common(3)],
        "avg_velocity":       round(float(np.mean([n["velocity"] for n in notes])), 1),
    }



# ─────────────────────────────────────────────
# 3. LEVEL 2 — RHYTHM
# ─────────────────────────────────────────────

def analyse_rhythm(notes: list, beats: dict, target_bpm: float = None) -> dict:
    """Tempo, tempo stability and hesitations."""
    if not notes:
        return {"status": "no_data"}
    onsets = [g[0]["start"] for g in rc.onset_groups(notes)]
    tempo = beats["tempo_bpm"]
    beat_sec = 60.0 / tempo if tempo else 1.0

    hesitations = [{"at": round(b, 2), "gap_sec": round(b - a, 2),
                    "gap_beats": round((b - a) / beat_sec, 1)}
                   for a, b in zip(onsets, onsets[1:]) if b - a > 2 * beat_sec]
    result = {
        "total_attacks":   len(onsets),
        "tempo_bpm":       tempo,
        "target_bpm":      target_bpm,
        "tempo_cv":        beats["tempo_cv"],
        "hesitation_count": len(hesitations),
        "hesitations":     hesitations[:5],
    }
    if target_bpm:
        result["tempo_vs_target"] = round(tempo / target_bpm - 1, 3)

    if len(onsets) < MIN_ATTACKS or len(beats["beat_times"]) < 3:
        result["timing_assessment"] = "Not enough notes to judge the tempo"
        return result

    cv = beats["tempo_cv"]
    result["timing_score"] = max(0, round(100 * (1 - cv / UNSTABLE_CV)))
    result["timing_assessment"] = ("Steady tempo" if cv < STEADY_CV else
                                   "Mostly steady tempo" if cv < MOSTLY_STEADY_CV else
                                   "The tempo varies a lot (rubato or unsteady pulse)")
    return result



# ─────────────────────────────────────────────
# 4. LEVEL 3 — HARMONY
# ─────────────────────────────────────────────

def analyse_harmony(chord_events: list) -> dict:
    """In-key share of the chords and known progressions (music21 Roman numerals)."""
    chords = [c for c in chord_events
              if c.get("fit") not in ("no_chord", "unknown", None)
              and c.get("chord", "N") not in ("N", "X")]
    if not chords:
        return {"status": "no_data",
                "note": "No chords detected. Run chord_analysis.py to enable harmony feedback."}

    counts = Counter(c["fit"] for c in chords)
    ratio = counts["in_scale"] / len(chords)
    if ratio >= 0.90:
        assessment = "Strong harmonic coherence: chords fit the key well."
    elif ratio >= 0.70:
        assessment = "Good harmony: a few chords outside the key."
    elif ratio >= 0.50:
        assessment = "Moderate harmony: several chords outside the key."
    else:
        assessment = "Many out-of-key chords."

    outside = Counter(c["chord"] for c in chords if c["fit"] == "out_of_scale")
    return {
        "total_chord_segments":  len(chords),
        "in_scale_count":        counts["in_scale"],
        "partial_count":         counts["partial"],
        "out_of_scale_count":    counts["out_of_scale"],
        "in_scale_ratio":        round(ratio, 3),
        "harmony_score":         round(ratio * 100),
        "harmony_assessment":    assessment,
        "non_diatonic_chords":   [{"chord": c, "count": n} for c, n in outside.most_common(3)],
        "progression":           " ".join(numeral_sequence(chords)[:16]),
        "progressions_detected": detect_progressions(numeral_sequence(chords)),
    }


def numeral_sequence(chords: list) -> list:
    """Roman numerals with consecutive repeats removed."""
    seq = []
    for c in chords:
        n = c.get("numeral")
        if n and (not seq or seq[-1] != n):
            seq.append(n)
    return seq


def detect_progressions(numerals: list) -> list:
    found = []
    for name, pattern in KNOWN_PROGRESSIONS.items():
        for i in range(len(numerals) - len(pattern) + 1):
            if numerals[i:i + len(pattern)] == pattern:
                found.append({"progression": name, "at_index": i})
    return found[:5]



# ─────────────────────────────────────────────
# 5. SUGGESTIONS
# ─────────────────────────────────────────────

def build_suggestions(note_fb: dict, rhythm_fb: dict, harmony_fb: dict) -> list:
    """Prioritised, actionable suggestions (priority 1 first)."""
    if note_fb.get("status") == "no_data":
        return [{"priority": 1, "area": "general",
                 "message": "No notes were detected. Check the microphone input level and play again."}]
    suggestions = []
    note_score = note_fb.get("score", 100)
    timing_score = rhythm_fb.get("timing_score")

    if note_score < 60:
        names = ", ".join(e["note"] for e in note_fb.get("frequent_errors", [])[:2])
        suggestions.append({"priority": 1, "area": "notes",
                            "message": f"Focus on note accuracy, especially {names}. "
                                       "Review which notes belong to the key."})
    elif note_score < 80:
        suggestions.append({"priority": 2, "area": "notes",
                            "message": "Good note accuracy overall; watch the occasional wrong note."})

    if timing_score is not None and timing_score < 50:
        tempo = rhythm_fb.get("target_bpm") or rhythm_fb.get("tempo_bpm") or 60
        suggestions.append({"priority": 1, "area": "rhythm",
                            "message": "The tempo varies a lot. If this wasn't intentional rubato, "
                                       "practise with a metronome "
                                       f"at {int(tempo * 0.7)} BPM (70% of the tempo)."})
    elif rhythm_fb.get("hesitation_count", 0) > 2:
        suggestions.append({"priority": 2, "area": "rhythm",
                            "message": f"There were {rhythm_fb['hesitation_count']} noticeable pauses. "
                                       "Try to keep the flow between phrases."})

    if harmony_fb.get("harmony_score", 100) < 50 and harmony_fb.get("non_diatonic_chords"):
        chords = ", ".join(c["chord"] for c in harmony_fb["non_diatonic_chords"][:2])
        suggestions.append({"priority": 2, "area": "harmony",
                            "message": f"Some chords don't fit the key ({chords}). "
                                       "Review the chords that belong to this key."})

    if note_score >= 90 and timing_score is not None and timing_score >= 80:
        suggestions.append({"priority": 3, "area": "encouragement",
                            "message": "Notes and rhythm are solid: you can focus on expression."})
    if not suggestions:
        suggestions.append({"priority": 3, "area": "general",
                            "message": "Keep practising regularly; consistency is key."})
    return sorted(suggestions, key=lambda s: s["priority"])



# ─────────────────────────────────────────────
# 6. OUTPUT WRITERS
# ─────────────────────────────────────────────

def global_score(feedback: dict):
    scores = [s for s in (feedback["note_accuracy"].get("score"),
                          feedback["rhythm"].get("timing_score"),
                          feedback["harmony"].get("harmony_score")) if s is not None]
    return round(sum(scores) / len(scores)) if scores else 0


def write_feedback_json(feedback: dict, base_path: Path) -> Path:
    out = base_path.parent / f"feedback_{base_path.stem}.json"
    out.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_feedback_txt(feedback: dict, base_path: Path) -> Path:
    out = base_path.parent / f"feedback_{base_path.stem}.txt"
    meta, notes = feedback["meta"], feedback["note_accuracy"]
    rhythm, harmony = feedback["rhythm"], feedback["harmony"]
    lines = ["=" * 60, " FEEDBACK REPORT ", f"Session : {base_path.name}",
             f"Key : {meta.get('scale', '?')} ({meta.get('scale_source', '?')})"
             f" | Tempo : {rhythm.get('tempo_bpm', '?')} BPM",
             "=" * 60, f"\n GLOBAL SCORE : {global_score(feedback)}/100\n",
             " ── Level 1: Notes ── "]
    if notes.get("status") == "no_data":
        lines.append("  No note data available")
    else:
        lines += [f"  Score       : {notes['score']}/100",
                  f"  Notes played: {notes['total_notes']}",
                  f"  In key      : {notes['in_scale_count']} ({round(notes['in_scale_ratio'] * 100)}%)",
                  f"  Assessment  : {notes['assessment']}"]
        if notes["frequent_errors"]:
            lines.append("  Out of key  : " + ", ".join(f"{e['note']} ({e['count']}x)"
                                                       for e in notes["frequent_errors"]))

    lines += ["", " ── Level 2: Rhythm ── "]
    if rhythm.get("status") == "no_data":
        lines.append("  No rhythm data available.")
    else:
        lines += [f"  Score       : {rhythm.get('timing_score', 'N/A')}/100",
                  f"  Assessment  : {rhythm.get('timing_assessment', '')}",
                  f"  Tempo       : {rhythm['tempo_bpm']} BPM"
                  + (f" (target {rhythm['target_bpm']})" if rhythm.get("target_bpm") else "")]
        if rhythm.get("hesitation_count"):
            lines.append(f"  Hesitations : {rhythm['hesitation_count']}")

    lines += ["", " ── Level 3: Harmony ── "]
    if harmony.get("status") == "no_data":
        lines.append("  No chord data — run chord_analysis.py first.")
    else:
        lines += [f"  Score       : {harmony['harmony_score']}/100",
                  f"  Assessment  : {harmony['harmony_assessment']}",
                  f"  Progression : {harmony['progression']}"]
        if harmony["non_diatonic_chords"]:
            lines.append("  Out of key  : " + ", ".join(f"{c['chord']} ({c['count']}x)"
                                                       for c in harmony["non_diatonic_chords"]))
        if harmony["progressions_detected"]:
            lines.append("  Known       : " + ", ".join(p["progression"]
                                                       for p in harmony["progressions_detected"]))

    lines += ["", " ── Suggestions ── "]
    lines += [f"{'  ★' if s['priority'] == 1 else '  ·'} [{s['area'].upper()}] {s['message']}"
              for s in feedback["suggestions"]]
    lines += ["", "=" * 60]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out



# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    session_path = Path(args.session)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    meta = session.get("meta", {})
    notes = rc.session_notes(session)

    key_name = meta.get("scale") or rc.detect_key(notes)["key"] or "C major"
    scale_pc = rc.scale_pitch_classes(key_name)
    wav = session_path.with_suffix(".wav")
    audio = librosa.load(str(wav), sr=rc.SAMPLE_RATE, mono=True)[0] if wav.exists() else None
    beats = rc.track_beats(notes=notes, audio=audio)
    target_bpm = meta.get("bpm") if meta.get("bpm_source") == "declared" else None

    chord_path = Path(args.chords) if args.chords else \
                 session_path.parent / f"{session_path.stem}_chords.json"
    chord_events = json.loads(chord_path.read_text(encoding="utf-8")).get("chords", []) \
                   if chord_path.exists() else []

    print("=" * 60)
    print("  Correction feedback")
    print(f"  Session : {session_path.name}")
    print(f"  Key     : {key_name}  |  Tempo: {beats['tempo_bpm']} BPM")
    print(f"  Notes   : {len(notes)}  |  Chords: {len(chord_events)} segments")
    print("=" * 60)

    note_fb = analyse_notes(notes, scale_pc)
    rhythm_fb = analyse_rhythm(notes, beats, target_bpm)
    harmony_fb = analyse_harmony(chord_events)
    feedback = {"meta": {**meta, "scale": key_name}, "note_accuracy": note_fb,
                "rhythm": rhythm_fb, "harmony": harmony_fb,
                "suggestions": build_suggestions(note_fb, rhythm_fb, harmony_fb)}

    print(f"\n  Global score : {global_score(feedback)}/100")
    print(f"  Notes        : {note_fb.get('assessment', note_fb.get('status'))}")
    print(f"  Rhythm       : {rhythm_fb.get('timing_assessment', rhythm_fb.get('status'))}")
    print(f"  Harmony      : {harmony_fb.get('harmony_assessment', harmony_fb.get('status'))}")
    for s in feedback["suggestions"]:
        print(f"{'  ★' if s['priority'] == 1 else '  ·'} {s['message']}")
    print(f"\n  JSON : {write_feedback_json(feedback, session_path)}")
    print(f"  TXT  : {write_feedback_txt(feedback, session_path)}")
    print("Done.")


if __name__ == "__main__":
    main()
