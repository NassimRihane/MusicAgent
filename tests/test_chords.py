import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

import chord_analysis as ca
import chord_model as cm
from conftest import SR, silence, tones, write_json

C_MAJOR = ca.scale_pcs_from_meta({"scale": "C major"})

# C – Am – F – G, 2 s each, after 1 s of silence
PROGRESSION = [("C", [48, 60, 64, 67]), ("A:min", [45, 57, 60, 64]),
               ("F", [41, 57, 60, 65]), ("G", [43, 59, 62, 67])]


@pytest.fixture(scope="module")
def progression_audio():
    return np.concatenate([silence(1.0)] + [tones(m, 2.0) for _, m in PROGRESSION])


def label_at(segments, t):
    return next(s["chord"] for s in segments if s["start"] <= t < s["end"])


# ─────────────────────────────────────────────
# Scale classification
# ─────────────────────────────────────────────

def test_chord_to_pitch_classes():
    assert ca.chord_to_pitch_classes("C") == {0, 4, 7}
    assert ca.chord_to_pitch_classes("A:min7") == {9, 0, 4, 7}
    assert ca.chord_to_pitch_classes("N") == set()


@pytest.mark.parametrize("label,fit", [
    ("C", "in_scale"), ("B:dim", "in_scale"), ("D", "partial"),
    ("F#", "out_of_scale"), ("N", "no_chord"),
])
def test_classify_chord_against_scale(label, fit):
    assert ca.classify_chord_against_scale(label, C_MAJOR) == fit


def test_scale_pcs_from_meta_defaults_to_c_major():
    assert ca.scale_pcs_from_meta({}) == C_MAJOR
    assert 6 in ca.scale_pcs_from_meta({"scale": "G major"})


def test_merge_consecutive():
    raw = [{"start": 0, "end": 1, "chord": "C", "score": 0.8},
           {"start": 1, "end": 2, "chord": "C", "score": 0.6},
           {"start": 2, "end": 3, "chord": "G", "score": 0.9}]
    merged = ca._merge_consecutive(raw)
    assert [(m["chord"], m["end"], m["score"]) for m in merged] == [("C", 2, 0.7), ("G", 3, 0.9)]


# ─────────────────────────────────────────────
# librosa backend
# ─────────────────────────────────────────────

def test_librosa_recognises_progression(progression_audio):
    segs = ca.analyse_librosa(progression_audio)
    # sample the middle of each chord, away from the transitions
    for i, (expected, _) in enumerate(PROGRESSION):
        assert label_at(segs, 1.0 + 2 * i + 1.0) == expected


def test_librosa_labels_silence_as_no_chord(progression_audio):
    assert label_at(ca.analyse_librosa(progression_audio), 0.3) == "N"


def test_librosa_segments_tile_without_overlap(progression_audio):
    segs = ca.analyse_librosa(progression_audio)
    for a, b in zip(segs, segs[1:]):
        assert a["end"] == pytest.approx(b["start"])


def test_auto_backend_uses_librosa_without_weights(progression_audio, monkeypatch, tmp_path):
    monkeypatch.setattr(cm, "WEIGHTS_PATH", tmp_path / "missing.npz")
    segs = ca.run_analysis(progression_audio[:SR * 3], "auto")
    assert {s["method"] for s in segs} == {"librosa"}


def test_main_writes_chord_files(tmp_path, progression_audio):
    wav = tmp_path / "session_test.wav"
    sf.write(wav, progression_audio, SR)
    write_json(tmp_path / "session_test.json", {"meta": {"scale": "C major"}, "events": []})

    ca.main([str(wav), "--backend", "librosa"])

    out = json.loads((tmp_path / "session_test_chords.json").read_text())
    fits = Counter(c["fit"] for c in out["chords"])
    assert fits["no_chord"] >= 1 and fits["in_scale"] >= 4
    assert (tmp_path / "session_test_chords.txt").exists()


# ─────────────────────────────────────────────
# BiLSTM model (mechanics only: it is untrained)
# ─────────────────────────────────────────────

def test_vocabulary():
    assert cm.N_CLASSES == 1 + 12 * 7
    assert cm.CHORD_VOCAB[0] == "N" and "A:min" in cm.CHORD_VOCAB


def test_template_matrix_rows_are_normalised():
    W = cm.build_template_matrix()
    assert np.allclose(W[0], 0)
    assert np.allclose(np.linalg.norm(W[1:], axis=1), 1)


def test_bilstm_forward_shape_and_probabilities():
    probs = cm.BiLSTMChordModel().forward(np.random.rand(7, 12).astype(np.float32))
    assert probs.shape == (7, cm.N_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1)


def test_bilstm_save_load_roundtrip(tmp_path):
    a, b = cm.BiLSTMChordModel(seed=1), cm.BiLSTMChordModel(seed=2)
    a.save(str(tmp_path / "w.npz"))
    b.load(str(tmp_path / "w.npz"))
    X = np.random.rand(5, 12).astype(np.float32)
    assert np.allclose(a.forward(X), b.forward(X))


def test_merge_short_segments():
    segs = [{"start": 0.0, "end": 1.0, "chord": "C"},
            {"start": 1.0, "end": 1.1, "chord": "D"},     # too short
            {"start": 1.1, "end": 2.0, "chord": "G"}]
    merged = cm._merge_short_segments(segs, 0.3)
    assert [s["chord"] for s in merged] == ["C", "G"]
    assert merged[0]["end"] == 1.1


def test_recogniser_predict_returns_segments(progression_audio):
    segs = cm.ChordRecogniser().predict(progression_audio[:SR * 3], SR)
    assert segs and all({"start", "end", "chord", "confidence", "method"} <= s.keys()
                        for s in segs)
