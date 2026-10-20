# chord_analysis.py
#
# Step 1c — Post-session chord analysis
#
# Reads a session WAV recording and returns a time-labelled chord sequence.
# Two backends are supported — both require only numpy and librosa:
#
#   bilstm
#     Built-in BiLSTM from chord_model.py.
#     Only meaningful once trained (chord_model_weights.npz): untrained,
#     its output is close to uniform noise.
#     No external dependencies beyond numpy and librosa.
#
#   librosa
#     Frame-by-frame chroma template matching.
#     No temporal context — each segment is classified independently.
#     Simpler, faster, less accurate on ambiguous chords.
#
#   auto (default) uses bilstm if trained weights exist, librosa otherwise.
#
# Output files (written next to the input):
#   session_XXX_chords.json   — machine-readable chord sequence
#   session_XXX_chords.txt    — human-readable summary
#
# Usage:
#   python chord_analysis.py sessions/session_20260101_120000.wav
#   python chord_analysis.py sessions/session_20260101_120000.json
#   python chord_analysis.py sessions/session_20260101_120000.wav --backend librosa

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import librosa


# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-session chord analysis (BiLSTM or librosa fallback)"
    )
    parser.add_argument("input",
        help="Path to a session WAV or JSON file")
    parser.add_argument("--backend", default="auto",
        choices=["auto", "bilstm", "librosa"],
        help="Analysis backend: 'auto' (default: bilstm if trained weights exist, "
             "else librosa), 'bilstm' or 'librosa' (template matching).")
    parser.add_argument("--segment-sec", type=float, default=0.5,
        help="Librosa backend: chroma window size in seconds (default 0.5)")
    parser.add_argument("--hop-sec", type=float, default=0.25,
        help="Librosa backend: hop between windows in seconds (default 0.25)")
    return parser.parse_args(argv)


# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

SAMPLE_RATE = 22050

# Windows quieter than this fraction of the loudest window are labelled "N"
# (no chord), so silence and room noise are not classified as chords.
SILENCE_REL_RMS = 0.05

CHORD_TYPE_INTERVALS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_NAME_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}


# ─────────────────────────────────────────────
# 2. CHORD TEMPLATES (librosa backend)
# ─────────────────────────────────────────────

def build_chord_templates() -> dict:
    templates = {}
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for chord_type, intervals in CHORD_TYPE_INTERVALS.items():
            label = root_name if chord_type == "maj" else f"{root_name}:{chord_type}"
            vec   = np.zeros(12, dtype=float)
            for interval in intervals:
                vec[(root_idx + interval) % 12] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            templates[label] = vec
    return templates

CHORD_TEMPLATES = build_chord_templates()


# ─────────────────────────────────────────────
# 3. AUDIO LOADING
# ─────────────────────────────────────────────

def load_audio(wav_path: Path) -> np.ndarray:
    print(f"  Loading audio: {wav_path.name}")
    audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    duration  = len(audio) / SAMPLE_RATE
    print(f"  Duration: {duration:.1f}s  |  Sample rate: {sr} Hz")
    return audio


# ─────────────────────────────────────────────
# 4. BACKEND A — BILSTM (chord_model.py)
# ─────────────────────────────────────────────

def analyse_bilstm(audio: np.ndarray) -> list:
    try:
        from chord_model import ChordRecogniser
    except ImportError:
        print("  ERROR: chord_model.py not found in the current directory.")
        raise
    recogniser = ChordRecogniser()
    return recogniser.predict(audio, SAMPLE_RATE)


# ─────────────────────────────────────────────
# 5. BACKEND B — LIBROSA TEMPLATE MATCHING
# ─────────────────────────────────────────────

def analyse_librosa(audio: np.ndarray, segment_sec: float = 0.5,
                    hop_sec: float = 0.25) -> list:
    print("  Running librosa template-matching (no temporal context)...")
    t0 = time.time()

    sr          = SAMPLE_RATE
    seg_samples = int(segment_sec * sr)
    hop_samples = int(hop_sec     * sr)
    events_raw  = []

    starts  = list(range(0, len(audio) - seg_samples, hop_samples))
    rms     = [float(np.sqrt(np.mean(audio[s : s + seg_samples] ** 2))) for s in starts]
    silence = max(rms, default=0.0) * SILENCE_REL_RMS

    for start_sample, seg_rms in zip(starts, rms):
        segment    = audio[start_sample : start_sample + seg_samples]
        start_time = start_sample / sr
        # Windows overlap; each one labels only its own hop so segments tile
        end_time   = (start_sample + hop_samples) / sr

        chroma      = librosa.feature.chroma_cqt(y=segment, sr=sr, bins_per_octave=36)
        chroma_mean = np.mean(chroma, axis=1)
        norm        = np.linalg.norm(chroma_mean)

        if norm < 1e-6 or seg_rms <= silence:
            events_raw.append({"start": round(start_time, 3),
                                "end":   round(end_time, 3),
                                "chord": "N", "score": 0.0})
            continue

        chroma_mean /= norm
        best_label, best_score = "N", -1.0
        for label, template in CHORD_TEMPLATES.items():
            score = float(np.dot(chroma_mean, template))
            if score > best_score:
                best_score = score
                best_label = label

        events_raw.append({"start": round(start_time, 3),
                            "end":   round(end_time, 3),
                            "chord": best_label,
                            "score": round(best_score, 3)})

    events  = _merge_consecutive(events_raw)
    elapsed = time.time() - t0
    print(f"  librosa finished in {elapsed:.1f}s — {len(events)} segments found")
    for e in events:
        e["method"] = "librosa"
    return events


def _merge_consecutive(events_raw: list) -> list:
    if not events_raw:
        return []
    merged  = []
    current = dict(events_raw[0])
    scores  = [current.get("score", 1.0)]
    for ev in events_raw[1:]:
        if ev["chord"] == current["chord"]:
            current["end"] = ev["end"]
            scores.append(ev.get("score", 1.0))
        else:
            current["score"] = round(float(np.mean(scores)), 3)
            merged.append(current)
            current = dict(ev)
            scores  = [current.get("score", 1.0)]
    current["score"] = round(float(np.mean(scores)), 3)
    merged.append(current)
    return merged


# ─────────────────────────────────────────────
# 6. BACKEND SELECTION
# ─────────────────────────────────────────────

def run_analysis(audio: np.ndarray, backend: str = "auto",
                 segment_sec: float = 0.5, hop_sec: float = 0.25) -> list:
    if backend == "auto":
        from chord_model import WEIGHTS_PATH
        backend = "bilstm" if WEIGHTS_PATH.exists() else "librosa"
    if backend == "bilstm":
        try:
            events = analyse_bilstm(audio)
            print(f"  BiLSTM analysis complete — {len(events)} segments found")
            return events
        except Exception as exc:
            print(f"  BiLSTM error: {exc}")
            print("  Falling back to librosa template-matching.")
    return analyse_librosa(audio, segment_sec, hop_sec)


# ─────────────────────────────────────────────
# 7. SCALE CLASSIFICATION HELPERS
# ─────────────────────────────────────────────

def chord_to_pitch_classes(chord_label: str) -> set:
    if chord_label in ("N", "X", ""):
        return set()
    if ":" in chord_label:
        root_str, quality = chord_label.split(":", 1)
    else:
        root_str, quality = chord_label, "maj"
    root_idx  = NOTE_NAMES.index(root_str) if root_str in NOTE_NAMES else 0
    intervals = CHORD_TYPE_INTERVALS.get(quality, CHORD_TYPE_INTERVALS["maj"])
    return {(root_idx + i) % 12 for i in intervals}


def classify_chord_against_scale(chord_label: str, scale_pcs: set) -> str:
    if chord_label in ("N", "X", ""):
        return "no_chord"
    chord_pcs = chord_to_pitch_classes(chord_label)
    if not chord_pcs:
        return "unknown"
    overlap = len(chord_pcs & scale_pcs) / len(chord_pcs)
    if overlap == 1.0:
        return "in_scale"
    elif overlap >= 0.5:
        return "partial"
    else:
        return "out_of_scale"


def scale_pcs_from_meta(meta: dict) -> set:
    scale_str = meta.get("scale", "C major").split()
    root      = scale_str[0] if scale_str else "C"
    mode      = scale_str[1] if len(scale_str) > 1 else "major"
    root_pc   = NOTE_NAME_TO_PC.get(root, 0)
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    return set((root_pc + i) % 12 for i in intervals)


# ─────────────────────────────────────────────
# 8. OUTPUT WRITERS
# ─────────────────────────────────────────────

def write_json(events: list, meta: dict, scale_pcs: set, base_path: Path) -> Path:
    enriched = []
    for e in events:
        entry        = dict(e)
        entry["fit"] = classify_chord_against_scale(e["chord"], scale_pcs)
        enriched.append(entry)
    output = {
        "meta":         meta,
        "chord_method": enriched[0]["method"] if enriched else "unknown",
        "chords":       enriched,
    }
    out_path = base_path.parent / f"{base_path.stem}_chords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return out_path


def write_txt(events: list, meta: dict, scale_pcs: set, base_path: Path) -> Path:
    out_path    = base_path.parent / f"{base_path.stem}_chords.txt"
    fit_markers = {"in_scale": "v", "partial": "~", "out_of_scale": "x",
                   "no_chord": " ", "unknown": "?"}
    lines = []
    lines.append("=" * 60)
    lines.append("  CHORD ANALYSIS REPORT")
    lines.append(f"  File   : {base_path.name}")
    lines.append(f"  Scale  : {meta.get('scale', '?')}")
    lines.append(f"  Method : {events[0]['method'] if events else '?'}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  {'START':>7}  {'END':>7}  {'CHORD':<14}  {'FIT':<14}  CONF")
    lines.append(f"  {'-'*7}  {'-'*7}  {'-'*14}  {'-'*14}  ----")

    for e in events:
        fit       = classify_chord_against_scale(e["chord"], scale_pcs)
        score     = e.get("score") or e.get("confidence")
        score_str = f"{score:.2f}" if isinstance(score, float) else "—"
        marker    = fit_markers.get(fit, " ")
        lines.append(f"  {e['start']:>7.2f}  {e['end']:>7.2f}  "
                     f"{e['chord']:<14}  {fit:<14}  {score_str}  {marker}")

    lines.append("")
    lines.append("  SUMMARY")
    lines.append(f"  Total chord segments : {len(events)}")
    chord_counts = Counter(e["chord"] for e in events)
    lines.append("  Most frequent chords :")
    for label, count in chord_counts.most_common(5):
        fit    = classify_chord_against_scale(label, scale_pcs)
        marker = fit_markers.get(fit, " ")
        lines.append(f"    {label:<14}  {count:>3}x   {marker} ({fit})")
    lines.append("=" * 60)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def print_timeline(events: list, scale_pcs: set):
    fit_markers = {"in_scale": "v", "partial": "~",
                   "out_of_scale": "x", "no_chord": " "}
    print("")
    print(f"  {'START':>7}  {'END':>7}  {'CHORD':<14}  FIT")
    print(f"  {'-'*7}  {'-'*7}  {'-'*14}  ---")
    for e in events:
        fit    = classify_chord_against_scale(e["chord"], scale_pcs)
        marker = fit_markers.get(fit, "?")
        print(f"  {e['start']:>7.2f}  {e['end']:>7.2f}  {e['chord']:<14}  {marker}")
    print("")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)

    if input_path.suffix.lower() == ".json":
        with open(input_path, encoding="utf-8") as f:
            session = json.load(f)
        meta     = session.get("meta", {})
        wav_path = input_path.with_suffix(".wav")
        if not wav_path.exists():
            print(f"  ERROR: WAV file not found: {wav_path}")
            print("  The WAV must have the same stem as the JSON file.")
            sys.exit(1)
    elif input_path.suffix.lower() == ".wav":
        wav_path  = input_path
        json_path = input_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f).get("meta", {})
        else:
            meta = {}
    else:
        print(f"  ERROR: unsupported file type '{input_path.suffix}'")
        sys.exit(1)

    scale_pcs = scale_pcs_from_meta(meta)

    print("=" * 60)
    print("  Chord Analysis")
    print(f"  Input  : {wav_path.name}")
    print(f"  Scale  : {meta.get('scale', 'unknown')}")
    print(f"  Backend: {args.backend}")
    print("=" * 60)

    audio  = load_audio(wav_path)
    events = run_analysis(audio, args.backend, args.segment_sec, args.hop_sec)

    if not events:
        print("  No chords detected.")
        sys.exit(0)

    print_timeline(events, scale_pcs)

    json_out = write_json(events, meta, scale_pcs, wav_path)
    txt_out  = write_txt(events,  meta, scale_pcs, wav_path)

    print(f"  JSON saved : {json_out}")
    print(f"  TXT  saved : {txt_out}")
    print("Done.")


if __name__ == "__main__":
    main()