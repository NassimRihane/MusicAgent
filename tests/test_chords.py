import json

import numpy as np
import pytest
import soundfile as sf

import chord_analysis as ca
import chord_rf
import recognition as rc
from conftest import SR, silence, tones, write_json

# C – Am – F – G, 2 s each, after 1 s of silence
PROGRESSION = [("C", [48, 60, 64, 67]), ("A:min", [45, 57, 60, 64]),
               ("F", [41, 57, 60, 65]), ("G", [43, 59, 62, 67])]
BEATS = [float(t) for t in np.arange(0, 9.0, 0.5)]


@pytest.fixture(scope="module")
def progression_audio():
    return np.concatenate([silence(1.0)] + [tones(m, 2.0) for _, m in PROGRESSION])


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    """A small Random Forest trained in the test (the real one lives in models/)."""
    return chord_rf.train(per_class=12, path=tmp_path_factory.mktemp("m") / "rf.joblib",
                          verbose=False)


def progression_session():
    notes = [{"start": 1.0 + 2 * i, "end": 3.0 + 2 * i, "midi": m, "velocity": 80}
             for i, (_, ms) in enumerate(PROGRESSION) for m in ms]
    return {"meta": {"scale": "C major"}, "notes": notes, "events": rc.notes_to_events(notes)}


def label_at(segments, t):
    return next(s["label"] for s in segments if s["start"] <= t < s["end"])


# ─────────────────────────────────────────────
# Random Forest chord model
# ─────────────────────────────────────────────

def test_vocabulary_matches_music21_labels():
    assert len(chord_rf.VOCAB) == 1 + 12 * 7
    assert chord_rf.VOCAB[0] == "N" and "A:min" in chord_rf.VOCAB and "G:dom7" in chord_rf.VOCAB
    for label in ("C", "A:min", "G:dom7", "B:dim", "F:maj7"):
        assert rc.chord_info(chord_rf.label_to_midis(label))["label"] == label


def test_synthetic_examples():
    rng = np.random.default_rng(0)
    y = chord_rf.synth_example("C", rng)
    assert len(y) == int(SR * chord_rf.CLIP_SEC) and np.abs(y).max() <= 0.1 + 1e-6
    assert np.abs(chord_rf.synth_example("N", rng)).max() < 0.05


def test_beat_chroma_shapes(progression_audio):
    feats, rms, bounds = chord_rf.beat_chroma(progression_audio, BEATS)
    assert feats.shape[1] == 12 and len(feats) == len(rms) == len(bounds) - 1
    assert np.allclose(np.linalg.norm(feats[rms > 0.01], axis=1), 1, atol=1e-3)


def test_training_saves_a_usable_model(small_model):
    assert small_model["heldout_accuracy"] > 0.5
    assert list(small_model["model"].classes_) == sorted(chord_rf.VOCAB)


def test_rf_recognises_progression(progression_audio, small_model):
    segs = chord_rf.predict_chords(progression_audio, BEATS, "C major", small_model)
    for i, (expected, _) in enumerate(PROGRESSION):
        assert label_at(segs, 1.0 + 2 * i + 1.0) == expected
    assert label_at(segs, 0.3) == "N"
    assert segs[1]["numeral"] == "I" and segs[1]["method"] == "rf"


def test_load_without_model_and_no_autotrain(tmp_path):
    with pytest.raises(FileNotFoundError):
        chord_rf.load(tmp_path / "missing.joblib", auto_train=False)


# ─────────────────────────────────────────────
# chord_analysis.py
# ─────────────────────────────────────────────

def test_music21_backend_from_notes():
    result = ca.run_analysis(progression_session(), None, "music21")
    assert result["key"] == "C major"
    assert [c["numeral"] for c in result["chords"] if c["chord"] != "N"] == ["I", "vi", "IV", "V"]


def test_rf_backend_falls_back_to_music21_without_audio(capsys):
    result = ca.run_analysis(progression_session(), None, "rf")
    assert result["chords"][0]["method"] == "music21"
    assert "using the music21 backend" in capsys.readouterr().out


def test_both_backends_report_agreement(progression_audio, small_model, monkeypatch):
    monkeypatch.setattr(chord_rf, "load", lambda *a, **k: small_model)
    result = ca.run_analysis(progression_session(), progression_audio, "both")
    assert result["chords"][0]["method"] == "rf"
    assert result["agreement"] > 0.8


def test_degree_agreement():
    a = [{"start": 0, "end": 2, "numeral": "I"}, {"start": 2, "end": 4, "numeral": "V"}]
    b = [{"start": 0, "end": 2, "numeral": "I"}, {"start": 2, "end": 4, "numeral": "IV"}]
    assert ca.degree_agreement(a, a) == 1.0
    assert ca.degree_agreement(a, b) == pytest.approx(0.5, abs=0.05)


def test_session_key_detected_for_older_sessions():
    # C – F – G – C: a progression that resolves, so the key is unambiguous
    notes = [{"start": i, "end": i + 1.0, "midi": m, "velocity": 80}
             for i, ms in enumerate([[48, 64, 67], [41, 57, 60], [43, 59, 62], [48, 64, 67]])
             for m in ms]
    assert ca.session_key({"meta": {}}, notes) == "C major"


def test_main_writes_chord_files(tmp_path, progression_audio, small_model, monkeypatch):
    monkeypatch.setattr(chord_rf, "load", lambda *a, **k: small_model)
    session = write_json(tmp_path / "session_t.json", progression_session())
    sf.write(tmp_path / "session_t.wav", progression_audio, SR)
    ca.main([str(session)])   # default backend: Random Forest
    out = json.loads((tmp_path / "session_t_chords.json").read_text())
    assert out["key"] == "C major" and out["chord_method"] == "rf"
    assert {c["fit"] for c in out["chords"]} <= {"in_scale", "no_chord"}
    assert "ROMAN" in (tmp_path / "session_t_chords.txt").read_text()


def test_main_rejects_audio_input(tmp_path):
    with pytest.raises(SystemExit):
        ca.main([str(tmp_path / "take.wav")])
