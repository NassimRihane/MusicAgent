# chord_rf.py
#
# Layer 3 — Random Forest chord recogniser (audio → chord labels)
#
# A standard machine-learning baseline for chord recognition: one 12-bin
# chroma vector per beat (librosa) is classified by a scikit-learn
# RandomForestClassifier into one of 85 chords (12 roots × 7 qualities, plus
# "N" for no chord).
#
# Training data is synthesised: piano-like chords in random voicings,
# inversions and octaves, sometimes with an added bass or melody note,
# rendered to audio and passed through the same chroma extraction. No
# chord-annotated piano recordings are freely available.
#
# This is the audio-based alternative to the default music21 chord naming
# (recognition.chords_from_notes); chord_analysis.py --backend rf uses it.
#
# Usage:
#   python chord_rf.py --train            # train and save models/chord_rf.joblib (~1 min)
#   python chord_rf.py --train --per-class 60


import argparse
import time
from pathlib import Path

import numpy as np
import librosa

import recognition as rc



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Random Forest chord recogniser")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--per-class", type=int, default=40,
                        help="Synthetic examples per chord class (default 40)")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. VOCABULARY
# ─────────────────────────────────────────────

QUALITIES = {
    "maj":  [0, 4, 7],     "min":  [0, 3, 7],     "dim": [0, 3, 6], "aug": [0, 4, 8],
    "dom7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10],
}

def label_for(root: int, quality: str) -> str:
    """Same labels as recognition.canonical_label: 'C', 'A:min', 'G:dom7'."""
    name = rc.NOTE_NAMES[root]
    return name if quality == "maj" else f"{name}:{quality}"

VOCAB = ["N"] + [label_for(r, q) for r in range(12) for q in QUALITIES]


def label_to_midis(label: str) -> list:
    """'A:min' → [57, 60, 64] (root position from A3); used to name chords with music21."""
    if label in ("N", "X"):
        return []
    root, _, quality = label.partition(":")
    pc = rc.NOTE_NAMES.index(root)
    return [57 - 9 + pc + i for i in QUALITIES[quality or "maj"]]



# ─────────────────────────────────────────────
# 2. FEATURES (librosa)
# ─────────────────────────────────────────────

SAMPLE_RATE = rc.SAMPLE_RATE
HOP = 512

def chroma(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """(12, frames) CQT chroma."""
    return librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP)


def beat_chroma(audio: np.ndarray, beat_times: list, sr: int = SAMPLE_RATE):
    """
    One L2-normalised chroma vector and one RMS value per beat interval
    (librosa.util.sync, median aggregation). Returns (features, rms, bounds).
    """
    C = chroma(audio, sr)
    rms = librosa.feature.rms(y=audio, hop_length=HOP)[0][:C.shape[1]]
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
    frames = np.unique(np.clip(frames, 0, C.shape[1]))
    sync = librosa.util.sync(C, frames, aggregate=np.median)
    sync_rms = librosa.util.sync(rms[None, :], frames, aggregate=np.mean)[0]
    bounds = np.concatenate([[0], frames, [C.shape[1]]])
    bounds = librosa.frames_to_time(np.unique(bounds), sr=sr, hop_length=HOP)
    feats = sync.T / (np.linalg.norm(sync.T, axis=1, keepdims=True) + 1e-9)
    return feats, sync_rms, bounds



# ─────────────────────────────────────────────
# 3. SYNTHETIC TRAINING DATA
# ─────────────────────────────────────────────

CLIP_SEC = 0.5

def piano_tone(midi: int, sr: int, dur: float, amp: float) -> np.ndarray:
    """Additive piano-like tone: decaying harmonics, short attack."""
    t = np.arange(int(sr * dur)) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = np.zeros_like(t)
    for k in range(1, 9):
        if f0 * k < sr / 2:
            y += np.sin(2 * np.pi * f0 * k * t) * np.exp(-t * (1.5 + 0.6 * k)) / k ** 1.2
    attack = np.minimum(1.0, t / 0.005)
    return amp * y * attack


def synth_example(label: str, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render one chord in a random voicing; 'N' is low noise."""
    n = int(sr * CLIP_SEC)
    if label == "N":
        return rng.normal(0, rng.uniform(1e-4, 3e-3), n).astype(np.float32)
    root, _, quality = label.partition(":")
    pc = rc.NOTE_NAMES.index(root)
    intervals = QUALITIES[quality or "maj"]
    base = 12 * rng.integers(3, 6) + pc                       # root in octave 2–4
    inversion = rng.integers(0, len(intervals))
    midis = [base + i + (12 if j < inversion else 0) for j, i in enumerate(intervals)]
    if rng.random() < 0.5:
        midis.append(base - 12)                               # bass doubling of the root
    if rng.random() < 0.3:
        midis.append(int(rng.choice(midis)) + 12)              # doubled chord tone above
    if rng.random() < 0.25:
        midis.append(int(base + 24 + rng.integers(0, 12)))     # passing melody note
    y = sum(piano_tone(m, sr, CLIP_SEC, rng.uniform(0.3, 1.0)) for m in midis)
    y += rng.normal(0, 1e-3, n)
    return (0.1 * y / max(np.abs(y).max(), 1e-9)).astype(np.float32)


def make_dataset(per_class: int = 40, seed: int = 0):
    """Synthesise per_class clips per label; one chroma vector per clip."""
    rng = np.random.default_rng(seed)
    labels = [lab for lab in VOCAB for _ in range(per_class)]
    audio = np.concatenate([synth_example(lab, rng) for lab in labels])
    bounds = [i * CLIP_SEC for i in range(1, len(labels))]
    X, _, _ = beat_chroma(audio, bounds)
    return X[:len(labels)], np.array(labels)



# ─────────────────────────────────────────────
# 4. TRAINING / LOADING
# ─────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "chord_rf.joblib"
N_JOBS     = -1   # all CPU cores

def train(per_class: int = 40, seed: int = 0, path: Path = MODEL_PATH, verbose: bool = True) -> dict:
    """Train on synthetic chords, report held-out accuracy, save the model."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    X, y = make_dataset(per_class, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
    rf.fit(Xtr, ytr)
    acc = float(rf.score(Xte, yte))
    rf.fit(X, y)   # final model on all examples

    bundle = {"model": rf, "vocab": VOCAB, "per_class": per_class, "seed": seed,
              "heldout_accuracy": round(acc, 3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    if verbose:
        print(f"  Chord RF: {len(y)} synthetic examples, held-out accuracy {acc:.1%} "
              f"({time.time() - t0:.0f}s) → {path}")
    return bundle


def load(path: Path = MODEL_PATH, auto_train: bool = True) -> dict:
    """Load the model; train it first if it doesn't exist yet."""
    import joblib
    if not path.exists():
        if not auto_train:
            raise FileNotFoundError(f"{path} not found. Run: python chord_rf.py --train")
        print("  No chord model yet — training it once (about a minute)...")
        return train(path=path)
    return joblib.load(path)



# ─────────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────────

SILENCE_REL_RMS = 0.05   # beats quieter than 5 % of the loudest are "N"

def predict_chords(audio: np.ndarray, beat_times: list, key_name: str = None,
                   bundle: dict = None, sr: int = SAMPLE_RATE) -> list:
    """Beat-synchronous chord segments: {start, end, chord, label, roman, numeral, fit}."""
    bundle = bundle or load()
    feats, rms, bounds = beat_chroma(audio, beat_times, sr)
    labels = bundle["model"].predict(feats)
    probs = bundle["model"].predict_proba(feats).max(axis=1)
    labels = np.where(rms <= SILENCE_REL_RMS * rms.max(), "N", labels)

    segments = []
    for i, lab in enumerate(labels):
        start, end = float(bounds[i]), float(bounds[i + 1])
        if segments and segments[-1]["label"] == lab:
            segments[-1]["end"] = end
            segments[-1]["_p"].append(probs[i])
            continue
        seg = {"start": start, "end": end, "label": str(lab), "_p": [probs[i]], "method": "rf"}
        if lab == "N":
            seg.update({"chord": "N", "fit": "no_chord"})
        else:
            info = rc.chord_info(label_to_midis(lab), key_name)
            seg.update({k: v for k, v in info.items() if k != "label"})
        segments.append(seg)
    for s in segments:
        s["confidence"] = round(float(np.mean(s.pop("_p"))), 3)
        s["start"], s["end"] = round(s["start"], 3), round(s["end"], 3)
    return segments



# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    if args.train:
        train(per_class=args.per_class)
    else:
        print("Nothing to do. Use --train to train the model.")


if __name__ == "__main__":
    main()
