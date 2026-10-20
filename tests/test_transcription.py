import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import transcription as tr
from conftest import SR, silence, tones

C_MAJOR = tr.build_scale("C", "major")


def raw(start, end, midi, velocity=80):
    return {"start": start, "end": end, "midi": midi, "velocity": velocity}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

@pytest.mark.parametrize("seconds,bpm,expected", [
    (1.0, 60, "quarter"), (0.5, 60, "eighth"), (2.0, 60, "half"),
    (1.5, 60, "dotted-quarter"), (0.5, 120, "quarter"), (0.33, 60, "eighth-triplet"),
])
def test_quantize_duration(seconds, bpm, expected):
    assert tr.quantize_duration(seconds, bpm) == expected


def test_note_names():
    assert tr.midi_to_note_name(60) == "C4"
    assert tr.midi_to_note_name(69) == "A4"
    assert tr.note_name_to_pc("Bb3") == 10


def test_buffer_rounds_time_and_writes_meta(tmp_path):
    buf = tr.TranscriptionBuffer("G minor", 80)
    buf.add({"time": 1.23456, "notes": []})
    data = buf.to_json(tmp_path / "s.json")
    assert data["events"][0]["time"] == 1.235
    saved = json.loads((tmp_path / "s.json").read_text())
    assert saved["meta"]["scale"] == "G minor" and saved["meta"]["bpm"] == 80


# ─────────────────────────────────────────────
# Event building
# ─────────────────────────────────────────────

def test_make_event_time_is_segment_start_plus_onset():
    ev = tr._make_event([raw(0.30, 1.30, 60)], 12.0, C_MAJOR, 60)
    assert ev["time"] == pytest.approx(12.30)
    assert ev["type"] == "note" and ev["rhythm"] == "quarter"


def test_make_event_merges_duplicate_pitches_keeping_loudest():
    group = [raw(0.10, 0.6, 64, 70), raw(0.12, 0.6, 60, 50), raw(0.13, 0.6, 60, 90)]
    ev = tr._make_event(group, 0.0, C_MAJOR, 60)
    assert [(n["name"], n["velocity"]) for n in ev["notes"]] == [("C4", 90), ("E4", 70)]
    assert ev["type"] == "chord"


def test_make_event_scale_flags():
    ev = tr._make_event([raw(0, 0.5, 60), raw(0, 0.5, 61)], 0.0, C_MAJOR, 60)
    assert [n["in_scale"] for n in ev["notes"]] == [True, False]
    assert ev["in_scale"] is False


def test_analyse_segment_groups_notes_with_fake_model(monkeypatch):
    import basic_pitch.inference as bpi
    notes = [NS(start=0.10, end=0.60, pitch=60, velocity=80),
             NS(start=0.15, end=0.60, pitch=64, velocity=70),   # within 80 ms → chord
             NS(start=1.10, end=1.50, pitch=67, velocity=60)]
    calls = []

    def fake_predict(path, model, **kw):
        calls.append(kw)
        return None, NS(instruments=[NS(notes=notes)]), None
    monkeypatch.setattr(bpi, "predict", fake_predict)
    monkeypatch.setattr(tr, "_get_basic_pitch_model", lambda: "model")

    events = tr.analyse_segment(np.zeros(SR, dtype=np.float32), 20.0, C_MAJOR, 90,
                                onset_thresh=0.4)
    assert [round(e["time"], 2) for e in events] == [20.10, 21.10]
    assert [len(e["notes"]) for e in events] == [2, 1]
    assert calls[0]["onset_threshold"] == 0.4 and calls[0]["midi_tempo"] == 90


def test_basic_pitch_model_loaded_once(monkeypatch):
    import basic_pitch.inference as bpi
    loads = []
    monkeypatch.setattr(bpi, "Model", lambda path: loads.append(path) or object())
    monkeypatch.setattr(tr, "_BP_MODEL", None)
    first = tr._get_basic_pitch_model()
    assert tr._get_basic_pitch_model() is first
    assert len(loads) == 1


# ─────────────────────────────────────────────
# Real model
# ─────────────────────────────────────────────

@pytest.mark.slow
def test_real_basic_pitch_transcribes_synthesised_notes(monkeypatch):
    monkeypatch.setattr(tr, "_BP_MODEL", None)
    audio = np.concatenate([silence(0.5), tones([60], 1.0, amp=0.3),
                            silence(0.5), tones([67], 1.0, amp=0.3)])
    events = tr.analyse_segment(audio, 10.0, C_MAJOR, 60)
    pitches = {n["midi"] for e in events for n in e["notes"]}
    assert {60, 67} <= pitches
    c4 = min(e["time"] for e in events if any(n["midi"] == 60 for n in e["notes"]))
    assert c4 == pytest.approx(10.5, abs=0.1)
