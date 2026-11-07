import json

import numpy as np
import pytest
import soundfile as sf

import expression_analysis as ea
import recognition as rc
from conftest import SR, silence, tones, write_json


def n(start, end, midi=60, velocity=80):
    return {"start": start, "end": end, "midi": midi, "velocity": velocity}


def test_articulation_staccato_vs_legato():
    staccato = rc.onset_groups([n(i, i + 0.3) for i in range(5)])
    legato = rc.onset_groups([n(i, i + 1.5) for i in range(5)])
    assert ea.articulation(staccato) == pytest.approx(0.3)
    assert ea.articulation(legato) == pytest.approx(1.5)


def test_timing_variability():
    even = rc.onset_groups([n(i, i + 0.5) for i in range(6)])
    uneven = rc.onset_groups([n(t, t + 0.5) for t in [0, 0.5, 2, 2.3, 4, 4.2]])
    assert ea.timing_variability(even) == pytest.approx(0)
    assert ea.timing_variability(uneven) > 0.5


def test_polyphony():
    assert ea.polyphony([n(0, 1, 60), n(0, 1, 64), n(0, 1, 67)]) == pytest.approx(3)
    assert ea.polyphony([n(0, 1, 60), n(1.05, 2, 62)]) == pytest.approx(1)


def test_note_features_complete_and_sensible():
    notes = [n(0.5 * i, 0.5 * i + 0.25, m, 60 + i) for i, m in enumerate([60, 64, 67, 72] * 4)]
    f = ea.note_features(notes)
    assert set(f) == set(ea.NOTE_FEATURES)
    assert f["tempo_bpm"] == pytest.approx(120, rel=0.05)
    assert f["onset_rate"] == pytest.approx(2.0, rel=0.1)
    assert f["articulation"] == pytest.approx(0.5)
    assert f["velocity_mean"] == pytest.approx(67.5)
    assert f["major_minor"] > 0


def test_note_features_need_two_notes():
    assert all(v is None for v in ea.note_features([n(0, 1)]).values())


def test_audio_features_louder_is_louder():
    quiet = ea.audio_features(tones([60, 64, 67], 2.0, amp=0.02))
    loud = ea.audio_features(tones([60, 64, 67], 2.0, amp=0.2))
    assert loud["loudness_db"] > quiet["loudness_db"] + 15
    bright = ea.audio_features(tones([84, 88, 91], 2.0))
    assert bright["brightness_hz"] > loud["brightness_hz"]


def happy_sad():
    happy = {"cues": {"performance_sec": 12, "tempo_bpm": 110, "loudness_db": -20,
                      "mean_pitch": 60, "dynamic_range_db": 10}}
    sad = {"cues": {"performance_sec": 16, "tempo_bpm": 100, "loudness_db": -20.5,
                    "mean_pitch": 60.3, "dynamic_range_db": 15}}
    return happy, sad


def test_compare_verdicts():
    happy, sad = happy_sad()
    rows = {r["cue"]: r["verdict"] for r in ea.compare(happy, sad, same_piece=True)}
    assert rows == {"performance_sec": "agrees", "loudness_db": "no clear change",
                    "mean_pitch": "no clear change", "dynamic_range_db": "opposite"}
    assert "tempo_bpm" in {r["cue"] for r in ea.compare(happy, sad)}


def test_load_input_session_with_wav(tmp_path):
    notes = [n(0, 1, 60), n(1, 2, 64)]
    path = write_json(tmp_path / "session_x.json", {"notes": notes})
    assert ea.load_input(path) == (notes, None)
    sf.write(tmp_path / "session_x.wav", tones([60], 1.0), SR)
    assert ea.load_input(path)[1] is not None


def test_main_compares_two_audio_files(tmp_path, monkeypatch):
    for name, length in (("a.wav", 2.0), ("b.wav", 3.0)):
        sf.write(tmp_path / name, np.concatenate([tones([60, 64, 67], length), silence(0.5)]), SR)
    fake = {"a.wav": [n(0.5 * i, 0.5 * i + 0.2, 60 + i % 3 * 4) for i in range(8)],
            "b.wav": [n(0.8 * i, 0.8 * i + 0.7, 57 + i % 3 * 3) for i in range(8)]}
    monkeypatch.setattr(rc, "transcribe", lambda path: fake[path.name])
    out = tmp_path / "result.json"
    ea.main([str(tmp_path / "a.wav"), "--compare", str(tmp_path / "b.wav"),
             "--same-piece", "--out", str(out)])
    result = json.loads(out.read_text())
    rows = {r["cue"]: r for r in result["comparison"]}
    assert rows["performance_sec"]["verdict"] == "agrees"      # b is slower
    assert rows["articulation"]["verdict"] == "agrees"         # b is more legato
    assert result["recordings"][1]["cues"]["key"] is not None
