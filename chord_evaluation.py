# chord_evaluation.py
#
# Layer 3 — Evaluation of the chord recognisers
#
# Measures how well the two chord backends recognise chords in realistic
# piano audio whose chords are known exactly:
#
#   test set   chord progressions in random keys and tempos, played in three
#              textures (block chords, arpeggios, chords + melody), rendered
#              with sampled piano sounds (FluidSynth + GeneralUser GS).
#              The Random Forest was trained on simple synthetic tones
#              (chord_rf.py), so this is a test on different audio.
#   backends   rf       Random Forest on beat-synchronous chroma (chord_rf.py)
#              music21  basic-pitch transcription → music21 chord naming
#              Both use librosa's beat tracking, as in the pipeline.
#   metrics    MIREX chord-estimation scores computed with mir_eval
#              (duration-weighted): root, majmin (root + major/minor),
#              sevenths (root + quality including 7ths).
#
# Output: models/chord_eval_report.json and models/chord_eval_report.md
#
# Real recordings (--recording): the learner plays the 28 chords of
# protocols/chord_recording.md, each struck once and held ~2 s. The take is
# split at the 28 strongest onsets (librosa), each chord is recognised by
# both backends and scored per chord with mir_eval.
# Output: models/chord_eval_recording_<take>.json / .md
#
# Usage:
#   python chord_evaluation.py                    # 36 pieces (12 per texture)
#   python chord_evaluation.py --pieces 60 --seed 1
#   python chord_evaluation.py --recording take.wav
#   python chord_evaluation.py --render-protocol expected.wav   # hear the expected take


import argparse
import json
import time
from pathlib import Path

import numpy as np

import chord_rf
import recognition as rc



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the chord recognisers")
    parser.add_argument("--pieces", type=int, default=36,
                        help="Number of test pieces, split evenly over the textures (default 36)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the test set")
    parser.add_argument("--recording", default=None,
                        help="Score a real recording of protocols/chord_recording.md instead")
    parser.add_argument("--render-protocol", default=None, metavar="WAV",
                        help="Render the chord protocol with the piano soundfont to this file")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

ROOT         = Path(__file__).parent
SOUNDFONT    = ROOT / "data" / "soundfonts" / "GeneralUser-GS.sf2"
SOUNDFONT_URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"
REPORT_JSON  = ROOT / "models" / "chord_eval_report.json"
REPORT_MD    = ROOT / "models" / "chord_eval_report.md"
PROTOCOL     = ROOT / "protocols" / "chord_recording.json"

TEXTURES = ["block", "arpeggio", "melody"]
METRICS  = ["root", "majmin", "sevenths"]
SAMPLE_RATE = rc.SAMPLE_RATE

# Common progressions as (scale degree in semitones, quality), per mode
PROGRESSIONS = {
    "major": [
        [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],                  # I V vi IV
        [(0, "maj"), (9, "min"), (5, "maj"), (7, "dom7")],                 # I vi IV V7
        [(2, "min7"), (7, "dom7"), (0, "maj7"), (0, "maj7")],              # ii7 V7 Imaj7
        [(0, "maj"), (5, "maj"), (11, "dim"), (4, "min"), (9, "min7"), (2, "min"), (7, "dom7"), (0, "maj")],
        [(0, "maj"), (0, "aug"), (9, "min"), (5, "maj7")],                 # I I+ vi IVmaj7
    ],
    "minor": [
        [(0, "min"), (5, "min"), (7, "dom7"), (0, "min")],                 # i iv V7 i
        [(0, "min"), (8, "maj"), (3, "maj"), (10, "maj")],                 # i VI III VII
        [(0, "min7"), (5, "min7"), (10, "dom7"), (3, "maj7")],             # i7 iv7 VII7 IIImaj7
        [(0, "min"), (10, "maj"), (8, "maj"), (7, "dom7")],                # i VII VI V7 (Andalusian)
        [(2, "dim"), (7, "dom7"), (0, "min"), (0, "min")],                 # ii° V7 i
    ],
}

HARTE = {"maj": "maj", "min": "min", "dim": "dim", "aug": "aug",
         "dom7": "7", "maj7": "maj7", "min7": "min7"}



# ─────────────────────────────────────────────
# 2. TEST SET
# ─────────────────────────────────────────────

def to_harte(label: str) -> str:
    """'A:min' → 'A:min', 'C' → 'C:maj', 'G:dom7' → 'G:7' (mir_eval syntax)."""
    if label in ("N", "X"):
        return label
    root, _, quality = label.partition(":")
    return f"{root}:{HARTE[quality or 'maj']}"


def chord_pitches(root_pc: int, quality: str) -> list:
    return [root_pc + i for i in chord_rf.QUALITIES[quality]]


def make_piece(rng: np.random.Generator, texture: str) -> dict:
    """
    One test piece: a progression in a random key and tempo, one chord per
    bar of 4 beats, repeated twice, played in `texture`.
    Returns notes, chord intervals and labels.
    """
    mode = str(rng.choice(["major", "minor"]))
    tonic = int(rng.integers(0, 12))
    progression = PROGRESSIONS[mode][rng.integers(0, len(PROGRESSIONS[mode]))] * 2
    beat = 60.0 / rng.uniform(70, 130)
    bar = 4 * beat
    start0 = 0.5
    notes, intervals, labels = [], [], []

    for i, (degree, quality) in enumerate(progression):
        t = start0 + i * bar
        root_pc = (tonic + degree) % 12
        pcs = chord_pitches(root_pc, quality)
        velocity = int(rng.integers(60, 100))
        bass = 36 + root_pc                                             # root in octave 2
        voicing = sorted(48 + (pc - 48) % 12 + 12 * (pc % 12 < 4) for pc in pcs)
        voicing = [p + 12 * int(rng.integers(0, 2)) for p in voicing]   # octave 3 or 4

        notes.append((bass, t, t + bar * 0.95, velocity))
        if texture == "block":
            for p in voicing:
                notes.append((p, t, t + bar * 0.95, velocity - 5))
        elif texture == "arpeggio":
            pattern = voicing + voicing[1:-1][::-1]                     # up and down
            step = beat / 2
            for k in range(8):
                p = pattern[k % len(pattern)]
                notes.append((p, t + k * step, t + (k + 1) * step, velocity - 10))
        else:  # melody: sustained chord + a melody of chord tones and passing tones
            for p in voicing:
                notes.append((p, t, t + bar * 0.95, velocity - 20))
            for k in range(4):
                tone = int(rng.choice(pcs)) if k % 2 == 0 else root_pc + int(rng.choice([2, 5, 9]))
                p = 72 + tone % 12
                notes.append((p, t + k * beat, t + (k + 1) * beat * 0.9, velocity + 5))

        intervals.append((t, t + bar))
        labels.append(to_harte(chord_rf.label_for(root_pc, quality)))

    return {"texture": texture, "key": f"{rc.NOTE_NAMES[tonic]} {mode}",
            "tempo": round(60 / beat, 1), "notes": notes,
            "intervals": np.array(intervals), "labels": labels}


def render(piece: dict, soundfont: Path = SOUNDFONT) -> np.ndarray:
    """Render a piece with FluidSynth (acoustic grand piano), peak-normalised to 0.5."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in piece["notes"]:
        inst.notes.append(pretty_midi.Note(int(np.clip(velocity, 1, 127)), int(pitch),
                                           float(start), float(end)))
    pm.instruments.append(inst)
    audio = pm.fluidsynth(fs=SAMPLE_RATE, synthesizer=str(soundfont))
    return (0.5 * audio / max(np.abs(audio).max(), 1e-9)).astype(np.float32)



# ─────────────────────────────────────────────
# 3. EVALUATION
# ─────────────────────────────────────────────

def estimate(audio: np.ndarray, backend: str, bundle: dict = None) -> tuple:
    """(intervals, Harte labels) estimated by one backend, as in the pipeline."""
    beats = rc.track_beats(audio=audio)["beat_times"]
    if backend == "rf":
        segs = chord_rf.predict_chords(audio, beats, bundle=bundle)
    else:
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            notes = rc.transcribe(tmp.name)
        segs = rc.chords_from_notes(notes, beats)
    if not segs:
        return np.array([[0.0, len(audio) / SAMPLE_RATE]]), ["N"]
    return (np.array([[s["start"], s["end"]] for s in segs]),
            [to_harte(s["label"]) for s in segs])


def score(ref_intervals, ref_labels, est_intervals, est_labels) -> dict:
    """MIREX chord scores (duration-weighted) with mir_eval."""
    import mir_eval
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(),
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    merged, ref, est = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(merged)
    result = {}
    for metric in METRICS:
        comparison = getattr(mir_eval.chord, metric)(ref, est)
        result[metric] = float(mir_eval.chord.weighted_accuracy(comparison, durations))
    return result


def evaluate(pieces: int = 36, seed: int = 0, bundle: dict = None, soundfont: Path = SOUNDFONT,
             verbose: bool = True) -> dict:
    """Generate, render and score the test set for both backends."""
    rng = np.random.default_rng(seed)
    bundle = bundle or chord_rf.load()
    rows, t0 = [], time.time()
    for i in range(pieces):
        piece = make_piece(rng, TEXTURES[i % len(TEXTURES)])
        audio = render(piece, soundfont)
        for backend in ("rf", "music21"):
            est = estimate(audio, backend, bundle)
            rows.append({"piece": i, "texture": piece["texture"], "key": piece["key"],
                         "tempo": piece["tempo"], "backend": backend,
                         **score(piece["intervals"], piece["labels"], *est)})
        if verbose and (i + 1) % 6 == 0:
            print(f"  {i + 1}/{pieces} pieces ({time.time() - t0:.0f}s)")
    return {"pieces": pieces, "seed": seed, "rows": rows, "summary": summarise(rows)}


def summarise(rows: list) -> dict:
    """Mean ± std of each metric per backend, overall and per texture."""
    summary = {}
    for backend in ("rf", "music21"):
        summary[backend] = {}
        for group in ["all"] + TEXTURES:
            sel = [r for r in rows if r["backend"] == backend and group in ("all", r["texture"])]
            if sel:
                summary[backend][group] = {
                    m: {"mean": round(float(np.mean([r[m] for r in sel])), 3),
                        "std": round(float(np.std([r[m] for r in sel])), 3)} for m in METRICS}
    return summary



# ─────────────────────────────────────────────
# 4. REPORT
# ─────────────────────────────────────────────

BACKEND_NAMES = {"rf": "Random Forest (audio chroma)",
                 "music21": "basic-pitch + music21 (notes)"}

def markdown_report(result: dict, heldout: float = None) -> str:
    s = result["summary"]
    lines = ["# Chord recognition — evaluation", "",
             f"{result['pieces']} test pieces (seed {result['seed']}): progressions in random keys "
             "and tempos, rendered with sampled piano sounds (FluidSynth, GeneralUser GS), in three "
             "textures. Beats from librosa, as in the pipeline. MIREX scores computed with mir_eval "
             "(duration-weighted), mean ± standard deviation over pieces.", ""]
    if heldout is not None:
        lines += [f"For reference, the Random Forest scores {heldout:.1%} on held-out *synthetic* "
                  "chords like those it was trained on.", ""]
    lines += ["| Recogniser | Texture | Root | Major/minor | Sevenths |", "|---|---|---|---|---|"]
    for backend in ("rf", "music21"):
        for group in ["all"] + TEXTURES:
            if group not in s.get(backend, {}):
                continue
            g = s[backend][group]
            lines.append(f"| {BACKEND_NAMES[backend] if group == 'all' else ''} | "
                         f"{'**all**' if group == 'all' else group} | "
                         + " | ".join(f"{g[m]['mean']:.1%} ± {g[m]['std']:.1%}" for m in METRICS) + " |")
    return "\n".join(lines) + "\n"


def write_reports(result: dict, heldout: float = None):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    REPORT_MD.write_text(markdown_report(result, heldout), encoding="utf-8")



# ─────────────────────────────────────────────
# 5. REAL RECORDINGS
# ─────────────────────────────────────────────

MIN_GAP_SEC   = 0.8   # two chords are at least this far apart
SILENCE_DB    = 35    # quieter than the loudest part by this much counts as silence
CHROMA_SEC    = 1.5   # the Random Forest looks at the first 1.5 s of each chord

def load_protocol(path: Path = PROTOCOL) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["chords"]


def render_protocol(out: Path, protocol: list = None, soundfont: Path = SOUNDFONT,
                    hold: float = 2.0, gap: float = 1.0) -> Path:
    """The expected take, rendered with the piano soundfont."""
    import soundfile as sf
    protocol = protocol or load_protocol()
    notes = [(p, 0.5 + i * (hold + gap), 0.5 + i * (hold + gap) + hold, 80)
             for i, c in enumerate(protocol) for p in c["pitches"]]
    sf.write(str(out), render({"notes": notes}, soundfont), SAMPLE_RATE)
    return Path(out)


def segment_chords(audio: np.ndarray, count: int) -> list:
    """
    Start time of each chord. The protocol leaves silence between chords, so
    the take is first split into non-silent stretches (librosa.effects.split).
    If that doesn't give `count` chords (e.g. pedal held), the `count`
    strongest onsets at least MIN_GAP_SEC apart are used instead.
    """
    import librosa
    regions = librosa.effects.split(audio, top_db=SILENCE_DB)
    starts, last_end = [], 0.0
    for begin, end in regions:   # merge stretches separated by very short gaps
        if starts and begin / SAMPLE_RATE - last_end < 0.25:
            last_end = end / SAMPLE_RATE
            continue
        starts.append(begin / SAMPLE_RATE)
        last_end = end / SAMPLE_RATE
    if len(starts) == count:
        return [round(float(t), 3) for t in starts]

    env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    wait = int(MIN_GAP_SEC * SAMPLE_RATE / 512)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SAMPLE_RATE, wait=wait)
    if len(frames) < count:
        raise ValueError(f"Found {max(len(starts), len(frames))} chords in the recording, "
                         f"expected {count}. Leave about 1 second of silence between chords.")
    strongest = sorted(frames[np.argsort(env[frames])[::-1][:count]])
    return [float(t) for t in librosa.frames_to_time(strongest, sr=SAMPLE_RATE)]


def recognise_segments(audio: np.ndarray, starts: list, bundle: dict) -> dict:
    """Harte labels per chord from both backends."""
    import tempfile, soundfile as sf
    C = chord_rf.chroma(audio)
    to_frame = lambda t: int(t * SAMPLE_RATE / chord_rf.HOP)
    ends = starts[1:] + [len(audio) / SAMPLE_RATE]
    feats = []
    for start, end in zip(starts, ends):
        v = np.median(C[:, to_frame(start) + 2:to_frame(min(end, start + CHROMA_SEC))], axis=1)
        feats.append(v / (np.linalg.norm(v) + 1e-9))
    rf = [to_harte(lab) for lab in bundle["model"].predict(np.array(feats))]

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        notes = rc.transcribe(tmp.name)
    m21 = []
    for start, end in zip(starts, ends):
        midis = [n["midi"] for n in notes if start - 0.05 <= n["start"] < end - 0.05]
        m21.append(to_harte(rc.chord_info(midis)["label"]) if midis else "N")
    return {"rf": rf, "music21": m21}


def score_per_chord(ref: list, est: list) -> dict:
    """Share of chords correct for each MIREX comparison (mir_eval); excluded chords skipped."""
    import mir_eval
    result = {}
    for metric in METRICS:
        comparison = np.array(getattr(mir_eval.chord, metric)(ref, est))
        valid = comparison >= 0
        result[metric] = round(float(comparison[valid].mean()), 3) if valid.any() else None
    return result


def evaluate_recording(audio: np.ndarray, protocol: list = None, bundle: dict = None) -> dict:
    protocol = protocol or load_protocol()
    bundle = bundle or chord_rf.load()
    starts = segment_chords(audio, len(protocol))
    est = recognise_segments(audio, starts, bundle)
    ref = [to_harte(c["label"]) for c in protocol]
    return {
        "chords": [{"number": c["number"], "name": c["name"], "expected": r, "start": round(t, 2),
                    "rf": a, "music21": b}
                   for c, r, t, a, b in zip(protocol, ref, starts, est["rf"], est["music21"])],
        "scores": {backend: score_per_chord(ref, est[backend]) for backend in ("rf", "music21")},
    }


def recording_report(result: dict, name: str) -> str:
    lines = [f"# Chord recognition — real recording `{name}`", "",
             f"{len(result['chords'])} chords of protocols/chord_recording.md, scored with mir_eval "
             "(share of chords correct; majmin ignores the augmented and diminished chords, "
             "sevenths ignores the augmented ones, as in MIREX).", "",
             "| Recogniser | Root | Major/minor | Sevenths |", "|---|---|---|---|"]
    for backend in ("rf", "music21"):
        sc = result["scores"][backend]
        lines.append(f"| {BACKEND_NAMES[backend]} | "
                     + " | ".join("—" if sc[m] is None else f"{sc[m]:.1%}" for m in METRICS) + " |")
    lines += ["", "| # | Expected | Random Forest | music21 |", "|---|---|---|---|"]
    for c in result["chords"]:
        mark = lambda e: e if e == c["expected"] else f"**{e}** ✗"
        lines.append(f"| {c['number']} | {c['expected']} | {mark(c['rf'])} | {mark(c['music21'])} |")
    return "\n".join(lines) + "\n"



# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    if args.recording:
        import librosa
        audio = librosa.load(args.recording, sr=SAMPLE_RATE, mono=True)[0]
        result = evaluate_recording(audio)
        stem = Path(args.recording).stem
        out = REPORT_JSON.parent / f"chord_eval_recording_{stem}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        report = recording_report(result, Path(args.recording).name)
        out.with_suffix(".md").write_text(report, encoding="utf-8")
        print(report)
        print(f"  Saved: {out.with_suffix('.md')}")
        return
    if not SOUNDFONT.exists():
        raise SystemExit(f"Soundfont missing: {SOUNDFONT}\n"
                         f"Download it (32 MB, free licence):\n"
                         f"  mkdir -p {SOUNDFONT.parent} && curl -L -o '{SOUNDFONT}' {SOUNDFONT_URL}")
    if args.render_protocol:
        print(f"  Saved: {render_protocol(Path(args.render_protocol))}")
        return
    bundle = chord_rf.load()
    result = evaluate(args.pieces, args.seed, bundle)
    write_reports(result, bundle.get("heldout_accuracy"))
    print()
    print(markdown_report(result, bundle.get("heldout_accuracy")))
    print(f"  Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
