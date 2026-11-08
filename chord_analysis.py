# chord_analysis.py
#
# Step 1c — Post-session chord analysis
#
# Labels the chords of a session, beat by beat, and places each one in the
# session's key (Roman numeral, in-key or not). Beats come from librosa's beat
# tracker on the session WAV (or on the note onsets if there is no WAV).
#
# Backends:
#   rf       (default)  Random Forest on beat-synchronous chroma of the audio
#                       (chord_rf.py); chords are then named and placed in the
#                       key by music21. Falls back to music21 without a WAV.
#   music21             chords named by music21 from the transcribed notes
#                       sounding in each beat — no machine learning
#   both                runs both, saves the Random Forest's result and
#                       reports how often the two agree on the scale degree
#
# The Random Forest is the default because it is clearly more accurate on
# realistic piano audio (chord_evaluation.py: 82 % vs 52 % majmin).
# basic-pitch shortens held notes as the piano sound decays, so music21 often
# names incomplete chords.
#
# Output files (written next to the input):
#   session_XXX_chords.json   — machine-readable chord sequence
#   session_XXX_chords.txt    — human-readable summary
#
# Usage:
#   python chord_analysis.py sessions/session_20260101_120000.json
#   python chord_analysis.py sessions/session_20260101_120000.json --backend music21
#   python chord_analysis.py sessions/session_20260101_120000.json --backend both

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import librosa

import recognition as rc


# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Post-session chord analysis")
    parser.add_argument("input", help="Path to a session JSON (its WAV is used if present)")
    parser.add_argument("--backend", default="rf", choices=["rf", "music21", "both"],
                        help="'rf' (default, Random Forest on audio), 'music21' (from notes) "
                             "or 'both' (compare them)")
    return parser.parse_args(argv)


# ─────────────────────────────────────────────
# 1. ANALYSIS
# ─────────────────────────────────────────────

def load_session(session_path: Path):
    """(session dict, audio or None) — the WAV with the same stem, if it exists."""
    session = json.loads(session_path.read_text(encoding="utf-8"))
    wav = session_path.with_suffix(".wav")
    audio = librosa.load(str(wav), sr=rc.SAMPLE_RATE, mono=True)[0] if wav.exists() else None
    return session, audio


def session_key(session: dict, notes: list) -> str:
    """The session's key (declared or detected), or detected now for older sessions."""
    return session.get("meta", {}).get("scale") or rc.detect_key(notes)["key"]


def run_analysis(session: dict, audio: np.ndarray = None, backend: str = "rf") -> dict:
    """Chord segments for a session, plus RF/music21 agreement with backend='both'."""
    notes = rc.session_notes(session)
    key_name = session_key(session, notes)
    beats = rc.track_beats(notes=notes, audio=audio)["beat_times"]

    if audio is None and backend != "music21":
        print("  No WAV for this session: using the music21 backend (from notes).")
        backend = "music21"
    result = {"key": key_name, "beats": len(beats)}
    if backend in ("rf", "both"):
        import chord_rf
        result["chords"] = chord_rf.predict_chords(audio, beats, key_name)
    if backend == "music21":
        result["chords"] = rc.chords_from_notes(notes, beats, key_name)
    if backend == "both":
        result["agreement"] = degree_agreement(result["chords"],
                                               rc.chords_from_notes(notes, beats, key_name))
    return result


def degree_agreement(a: list, b: list, step: float = 0.1) -> float:
    """
    Share of time (sampled every `step` s, chords only) where two chord
    sequences give the same scale degree (Roman numeral, case and sevenths ignored).
    """
    def at(segs, t):
        for s in segs:
            if s["start"] <= t < s["end"]:
                return (s.get("numeral") or "N").upper()
        return None
    if not a or not b:
        return 0.0
    ts = np.arange(max(a[0]["start"], b[0]["start"]), min(a[-1]["end"], b[-1]["end"]), step)
    pairs = [(at(a, t), at(b, t)) for t in ts]
    pairs = [(x, y) for x, y in pairs if x not in (None, "N") and y not in (None, "N")]
    return round(float(np.mean([x == y for x, y in pairs])), 3) if pairs else 0.0


# ─────────────────────────────────────────────
# 2. OUTPUT WRITERS
# ─────────────────────────────────────────────

FIT_MARKERS = {"in_scale": "v", "partial": "~", "out_of_scale": "x", "no_chord": " "}

def write_json(result: dict, meta: dict, base_path: Path) -> Path:
    chords = result["chords"]
    output = {"meta": meta, "key": result["key"],
              "chord_method": chords[0]["method"] if chords else "unknown",
              "agreement_rf_music21": result.get("agreement"),
              "chords": chords}
    out_path = base_path.parent / f"{base_path.stem}_chords.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_txt(result: dict, base_path: Path) -> Path:
    chords = result["chords"]
    out_path = base_path.parent / f"{base_path.stem}_chords.txt"
    lines = ["=" * 60, "  CHORD ANALYSIS REPORT", f"  File   : {base_path.name}",
             f"  Key    : {result['key']}",
             f"  Method : {chords[0]['method'] if chords else '?'}", "=" * 60, "",
             f"  {'START':>7}  {'END':>7}  {'CHORD':<22}  {'ROMAN':<7}  FIT",
             f"  {'-'*7}  {'-'*7}  {'-'*22}  {'-'*7}  ---"]
    for c in chords:
        lines.append(f"  {c['start']:>7.2f}  {c['end']:>7.2f}  {c['chord'][:22]:<22}  "
                     f"{c.get('roman', ''):<7}  {FIT_MARKERS.get(c['fit'], '?')}")
    lines += ["", "  SUMMARY", f"  Chord segments : {len(chords)}", "  Most frequent  :"]
    for label, count in Counter(c["chord"] for c in chords if c["chord"] != "N").most_common(5):
        lines.append(f"    {label:<22} {count:>3}x")
    if result.get("agreement") is not None:
        lines.append(f"  RF / music21 agreement on scale degree: {result['agreement']:.0%}")
    lines.append("=" * 60)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    session_path = Path(args.input)
    if session_path.suffix.lower() != ".json":
        print("  ERROR: give the session JSON (run transcription.py --audio FILE first).")
        sys.exit(1)
    session, audio = load_session(session_path)

    print("=" * 60)
    print("  Chord Analysis")
    print(f"  Input   : {session_path.name}  ({'with' if audio is not None else 'no'} WAV)")
    print(f"  Backend : {args.backend}")
    print("=" * 60)

    result = run_analysis(session, audio, args.backend)
    if not result["chords"]:
        print("  No chords detected.")
        sys.exit(0)
    print(f"  Key {result['key']} — {len(result['chords'])} chord segments")
    print("  " + " ".join(f"{c['chord']}" for c in result["chords"] if c["chord"] != "N")[:300])
    if result.get("agreement") is not None:
        print(f"  RF / music21 agreement on scale degree: {result['agreement']:.0%}")

    print(f"  JSON saved : {write_json(result, session.get('meta', {}), session_path)}")
    print(f"  TXT  saved : {write_txt(result, session_path)}")
    print("Done.")


if __name__ == "__main__":
    main()
