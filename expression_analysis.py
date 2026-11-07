# expression_analysis.py
#
# Layer 2 — Expressive descriptors
#
# Measures the performance cues that music-emotion research links to
# perceived emotion (Juslin & Laukka 2003; Gabrielsson & Lindström 2010).
# All are standard descriptors computed with standard tools:
#
#   from the notes (recognition.py: basic-pitch, music21, librosa)
#     tempo_bpm           librosa beat tracking on note onsets
#     onset_rate          attacks per second
#     timing_variability  coefficient of variation of inter-onset intervals
#     articulation        median of (sounding duration / time to next attack):
#                         < 1 detached (staccato), > 1 overlapping (legato, pedal)
#     velocity_mean/std   loudness of the notes (MIDI velocity, 0–127)
#     mean_pitch          register (MIDI number)
#     pitch_range         5th–95th percentile range, in semitones
#     polyphony           mean number of notes sounding at once
#     major_minor         music21 key analysis: major − minor correlation
#
#   from the audio (librosa)
#     loudness_db, dynamic_range_db, brightness_hz (spectral centroid)
#
# The note features are computed the same way on EMOPIA's MIDI files and on
# our transcriptions, so the emotion model (emotion_model.py) can use them.
#
# With two recordings (--compare), it reports how the second differs from the
# first and checks each difference against the direction expected for a
# happy → sad change. With --same-piece, tempo is compared by how long each
# performance takes, which is more robust than beat tracking under rubato.
#
# Usage:
#   python expression_analysis.py happy.mp3
#   python expression_analysis.py sessions/session_X.json
#   python expression_analysis.py happy.mp3 --compare sad.mp3 --same-piece --out results.json


import argparse
import json
from pathlib import Path

import numpy as np
import librosa

import recognition as rc



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Expressive descriptors of a performance")
    parser.add_argument("input", help="Audio file or session JSON")
    parser.add_argument("--compare", default=None,
                        help="Second recording to compare against the first")
    parser.add_argument("--same-piece", action="store_true",
                        help="Both recordings play the same piece: compare tempo by duration")
    parser.add_argument("--out", default=None, help="Output JSON path")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

SILENCE_DB = -50.0   # audio frames quieter than this are ignored for loudness

# Features used by the emotion model (note-level only: EMOPIA provides MIDI, not audio)
NOTE_FEATURES = ["tempo_bpm", "onset_rate", "timing_variability", "articulation",
                 "velocity_mean", "velocity_std", "mean_pitch", "pitch_range",
                 "polyphony", "major_minor"]

# Expected direction of each cue when going from a happy to a sad performance
# (+1: sad is higher, -1: sad is lower), after Juslin & Laukka (2003).
SAD_VS_HAPPY = {
    "tempo_bpm":          (-1, "Sad is slower"),
    "performance_sec":    (+1, "Sad is slower (same melody takes longer)"),
    "major_minor":        (-1, "Sad tends to minor mode"),
    "loudness_db":        (-1, "Sad is softer"),
    "articulation":       (+1, "Sad is more legato"),
    "mean_pitch":         (-1, "Sad uses a lower register"),
    "brightness_hz":      (-1, "Sad has a darker timbre"),
    "timing_variability": (+1, "Sad has more rubato / timing variation"),
    "dynamic_range_db":   (-1, "Happy has more contrast between soft and loud"),
}

# Smallest change counted as a real difference
MIN_CHANGE = {
    "tempo_bpm": 3.0, "performance_sec": 0.5, "major_minor": 0.05,
    "loudness_db": 1.0, "articulation": 0.1, "mean_pitch": 1.0,
    "brightness_hz": 50.0, "timing_variability": 0.05, "dynamic_range_db": 1.0,
}



# ─────────────────────────────────────────────
# 2. DESCRIPTORS
# ─────────────────────────────────────────────

def articulation(groups: list) -> float:
    """Median of (longest sounding duration in an attack / time to the next attack)."""
    ratios = []
    for g, nxt in zip(groups, groups[1:]):
        ioi = nxt[0]["start"] - g[0]["start"]
        if ioi > 0:
            ratios.append(max(n["end"] - n["start"] for n in g) / ioi)
    return float(np.median(ratios)) if ratios else 0.0


def timing_variability(groups: list) -> float:
    """Coefficient of variation of inter-onset intervals (higher = less even)."""
    iois = np.diff([g[0]["start"] for g in groups])
    iois = iois[iois > 0]
    return float(np.std(iois) / np.mean(iois)) if len(iois) > 1 else 0.0


def polyphony(notes: list, step: float = 0.05) -> float:
    """Mean number of notes sounding at once, over the time something sounds."""
    end = max(n["end"] for n in notes)
    counts = np.zeros(int(end / step) + 1)
    for n in notes:
        counts[int(n["start"] / step):int(n["end"] / step) + 1] += 1
    sounding = counts[counts > 0]
    return float(sounding.mean()) if len(sounding) else 0.0


def note_features(notes: list) -> dict:
    """The NOTE_FEATURES of a list of notes (None values if there are no notes)."""
    if len(notes) < 2:
        return {f: None for f in NOTE_FEATURES}
    groups = rc.onset_groups(notes)
    span = max(groups[-1][0]["start"] - groups[0][0]["start"], 1e-3)
    pitches = np.array([n["midi"] for n in notes])
    velocities = np.array([n["velocity"] for n in notes], dtype=float)
    return {
        "tempo_bpm":          rc.track_beats(notes=notes)["tempo_bpm"],
        "onset_rate":         round(len(groups) / span, 3),
        "timing_variability": round(timing_variability(groups), 3),
        "articulation":       round(articulation(groups), 3),
        "velocity_mean":      round(float(velocities.mean()), 2),
        "velocity_std":       round(float(velocities.std()), 2),
        "mean_pitch":         round(float(pitches.mean()), 2),
        "pitch_range":        float(np.percentile(pitches, 95) - np.percentile(pitches, 5)),
        "polyphony":          round(polyphony(notes), 3),
        "major_minor":        rc.detect_key(notes)["major_minor"],
    }


def audio_features(y: np.ndarray, sr: int = rc.SAMPLE_RATE) -> dict:
    """Loudness, dynamic range and brightness from the audio (librosa)."""
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=1.0)
    db = db[db > SILENCE_DB]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return {
        "loudness_db":      round(float(np.mean(db)), 1) if len(db) else None,
        "dynamic_range_db": round(float(np.percentile(db, 95) - np.percentile(db, 5)), 1)
                            if len(db) else None,
        "brightness_hz":    round(float(np.mean(centroid)), 0),
    }


def describe(notes: list, audio: np.ndarray = None, sr: int = rc.SAMPLE_RATE) -> dict:
    """All cues of one performance."""
    groups = rc.onset_groups(notes)
    duration = len(audio) / sr if audio is not None else \
               max((n["end"] for n in notes), default=0.0)
    cues = {
        "duration_sec":    round(duration, 2),
        "performance_sec": round(groups[-1][0]["start"] - groups[0][0]["start"], 2)
                           if groups else None,
        "key":             rc.detect_key(notes)["key"],
        "note_count":      len(notes),
        **note_features(notes),
    }
    if audio is not None:
        cues.update(audio_features(audio, sr))
    return cues


def load_input(path) -> tuple:
    """(notes, audio or None) from an audio file or a session JSON (with its WAV if present)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        session = json.loads(path.read_text(encoding="utf-8"))
        wav = path.with_suffix(".wav")
        audio = librosa.load(str(wav), sr=rc.SAMPLE_RATE, mono=True)[0] if wav.exists() else None
        return rc.session_notes(session), audio
    audio = librosa.load(str(path), sr=rc.SAMPLE_RATE, mono=True)[0]
    return rc.transcribe(path), audio


def analyse(path) -> dict:
    notes, audio = load_input(path)
    return {"file": Path(path).name, "cues": describe(notes, audio), "notes": notes}



# ─────────────────────────────────────────────
# 3. COMPARISON
# ─────────────────────────────────────────────

def compare(happy: dict, sad: dict, same_piece: bool = False) -> list:
    """
    For each cue, is the happy → sad change in the direction research expects?
    verdict: "agrees", "opposite", or "no clear change" (below MIN_CHANGE).
    """
    skip = "tempo_bpm" if same_piece else "performance_sec"
    rows = []
    for cue, (direction, expectation) in SAD_VS_HAPPY.items():
        a, b = happy["cues"].get(cue), sad["cues"].get(cue)
        if cue == skip or a is None or b is None:
            continue
        diff = b - a
        if abs(diff) < MIN_CHANGE[cue]:
            verdict = "no clear change"
        else:
            verdict = "agrees" if np.sign(diff) == direction else "opposite"
        rows.append({"cue": cue, "first": a, "second": b,
                     "change": round(diff, 3), "expected": expectation,
                     "verdict": verdict})
    return rows



# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    first = analyse(args.input)
    result = {"recordings": [first]}
    if args.compare:
        second = analyse(args.compare)
        result["recordings"].append(second)
        result["comparison"] = compare(first, second, args.same_piece)

    out = Path(args.out) if args.out else \
          Path(args.input).with_name(f"expression_{Path(args.input).stem}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for rec in result["recordings"]:
        print(f"\n  {rec['file']}")
        for k, v in rec["cues"].items():
            print(f"    {k:<20} {v}")
    marks = {"agrees": "✓", "opposite": "✗", "no clear change": "·"}
    for row in result.get("comparison", []):
        print(f"  {marks[row['verdict']]} {row['cue']:<20} {row['first']} → {row['second']}"
              f"   ({row['expected']})")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
