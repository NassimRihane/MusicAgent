from types import SimpleNamespace as NS

import numpy as np
import pytest

import recognition as rc
from conftest import SR, silence, tones


def n(start, end, midi, velocity=80):
    return {"start": start, "end": end, "midi": midi, "velocity": velocity}


def triad_notes(midis, t0, dur=1.0):
    return [n(t0, t0 + dur, m) for m in midis]


# ─────────────────────────────────────────────
# Notes
# ─────────────────────────────────────────────

def test_note_name():
    assert rc.note_name(60) == "C4" and rc.note_name(69) == "A4"


@pytest.mark.parametrize("seconds,bpm,expected", [
    (1.0, 60, "quarter"), (0.5, 60, "eighth"), (1.5, 60, "dotted-quarter"),
    (0.5, 120, "quarter"), (0.33, 60, "eighth-triplet"),
])
def test_quantize_duration(seconds, bpm, expected):
    assert rc.quantize_duration(seconds, bpm) == expected


def test_onset_groups():
    groups = rc.onset_groups([n(0, 1, 60), n(0.03, 1, 64), n(1, 2, 67)])
    assert [len(g) for g in groups] == [2, 1]


def test_notes_to_events_merges_duplicates_and_flags_scale():
    notes = [n(0.10, 0.6, 64, 70), n(0.12, 0.6, 60, 50), n(0.13, 0.6, 60, 90), n(1.0, 2.0, 61)]
    events = rc.notes_to_events(notes, rc.scale_pitch_classes("C major"), 60)
    assert [(x["name"], x["velocity"]) for x in events[0]["notes"]] == [("C4", 90), ("E4", 70)]
    assert events[0]["type"] == "chord" and events[0]["time"] == 0.1
    assert events[1]["in_scale"] is False and events[1]["rhythm"] == "quarter"


def test_notes_to_events_without_scale():
    assert rc.notes_to_events([n(0, 1, 60)])[0]["in_scale"] is None


def test_session_notes_prefers_notes_then_rebuilds_from_events():
    assert rc.session_notes({"notes": [n(0, 1, 60)]}) == [n(0, 1, 60)]
    legacy = {"events": [{"time": 2.0, "duration": 0.5,
                          "notes": [{"midi": 60, "confidence": 0.5}]}]}
    assert rc.session_notes(legacy) == [{"start": 2.0, "end": 2.5, "midi": 60, "velocity": 63.5}]


# ─────────────────────────────────────────────
# MIDI reading (pretty_midi) and the sustain pedal
# ─────────────────────────────────────────────

def fake_midi(notes, pedal=()):
    inst = NS(is_drum=False,
              notes=[NS(start=s, end=e, pitch=p, velocity=v) for s, e, p, v in notes],
              control_changes=[NS(number=64, time=t, value=v) for t, v in pedal])
    return NS(instruments=[inst])


def test_midi_to_notes_without_pedal():
    notes = rc.midi_to_notes(fake_midi([(0, 0.5, 60, 80)]))
    assert notes == [{"start": 0.0, "end": 0.5, "midi": 60, "velocity": 80}]


def test_sustain_pedal_extends_notes_until_release():
    pm = fake_midi([(0, 0.5, 60, 80), (1.0, 1.2, 64, 80)], pedal=[(0.2, 127), (2.0, 0)])
    assert [x["end"] for x in rc.midi_to_notes(pm, sustain=True)] == [2.0, 2.0]


def test_restruck_note_ends_previous_one():
    pm = fake_midi([(0, 0.5, 60, 80), (1.0, 1.2, 60, 80)], pedal=[(0.2, 127), (3.0, 0)])
    assert [x["end"] for x in rc.midi_to_notes(pm, sustain=True)] == [1.0, 3.0]


# ─────────────────────────────────────────────
# Key and scale (music21)
# ─────────────────────────────────────────────

def test_scale_pitch_classes():
    assert rc.scale_pitch_classes("C major") == {0, 2, 4, 5, 7, 9, 11}
    assert rc.scale_pitch_classes("A minor") == {9, 11, 0, 2, 4, 5, 7, 6, 8}   # + F#, G#
    assert 10 in rc.scale_pitch_classes("Bb major")


@pytest.mark.parametrize("triads,key", [
    ([[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]], "C major"),
    ([[57, 60, 64], [62, 65, 69], [64, 68, 71], [57, 60, 64]], "A minor"),
])
def test_detect_key(triads, key):
    notes = [x for i, t in enumerate(triads) for x in triad_notes(t, i)]
    result = rc.detect_key(notes)
    assert result["key"] == key
    assert (result["major_minor"] > 0) == key.endswith("major")


def test_detect_key_no_notes():
    assert rc.detect_key([])["key"] is None


# ─────────────────────────────────────────────
# Beats and tempo (librosa)
# ─────────────────────────────────────────────

def test_track_beats_from_steady_notes():
    notes = [n(t, t + 0.2, 60) for t in np.arange(0, 16, 0.5)]   # 120 BPM
    beats = rc.track_beats(notes=notes)
    assert beats["tempo_bpm"] == pytest.approx(120, rel=0.05)
    assert beats["tempo_cv"] < 0.01


def test_tempo_cv_grows_with_timing_jitter():
    rng = np.random.default_rng(0)
    steady = np.arange(0, 16, 0.3)
    cv = [rc.track_beats(notes=[n(t, t + 0.2, 60) for t in steady + rng.normal(0, j, len(steady))])
          ["tempo_cv"] for j in (0.0, 0.01, 0.04)]
    assert cv[0] < cv[1] < cv[2]


def test_track_beats_from_audio():
    clicks = np.concatenate([np.concatenate([tones([72], 0.05), silence(0.45)])
                             for _ in range(24)])
    beats = rc.track_beats(audio=clicks)
    assert beats["tempo_bpm"] == pytest.approx(120, rel=0.05)


# ─────────────────────────────────────────────
# Chords (music21)
# ─────────────────────────────────────────────

@pytest.mark.parametrize("midis,symbol,label,numeral", [
    ([48, 64, 67], "C", "C", "I"),
    ([52, 60, 67], "C/E", "C", "I"),
    ([43, 59, 62, 65], "G7", "G:dom7", "V"),
    ([45, 60, 64], "Am", "A:min", "vi"),
    ([47, 62, 65], "Bdim", "B:dim", "vii"),
])
def test_chord_info(midis, symbol, label, numeral):
    info = rc.chord_info(midis, "C major")
    assert (info["chord"], info["label"], info["numeral"]) == (symbol, label, numeral)
    assert info["fit"] == "in_scale"


def test_chord_info_out_of_key():
    assert rc.chord_info([54, 58, 61], "C major")["fit"] == "out_of_scale"   # F# major


def test_chords_from_notes_follows_a_progression():
    progression = [[48, 64, 67], [45, 60, 64], [41, 57, 60], [43, 59, 62]]   # C Am F G
    notes = [x for i, t in enumerate(progression) for x in triad_notes(t, i * 2.0, 2.0)]
    beats = list(np.arange(0, 8.0, 1.0))
    segs = rc.chords_from_notes(notes, beats, "C major")
    assert [s["numeral"] for s in segs] == ["I", "vi", "IV", "V"]
    assert [s["start"] for s in segs] == [0.0, 2.0, 4.0, 6.0]


def test_chords_from_notes_extends_two_note_beats():
    # beat 1 has only C and G; with beat 2 (E) the window holds a C major triad
    notes = [n(0, 0.9, 48), n(0, 0.9, 55), n(1.0, 1.9, 64)]
    segs = rc.chords_from_notes(notes, [0.0, 1.0, 2.0, 3.0], "C major")
    assert segs[0]["label"] == "C" and segs[0]["end"] == 2.0


def test_chords_from_notes_marks_silence():
    segs = rc.chords_from_notes(triad_notes([48, 64, 67], 0, 1.0), [0.0, 1.0, 2.0, 3.0], "C major")
    assert segs[-1]["chord"] == "N" and segs[-1]["fit"] == "no_chord"


# ─────────────────────────────────────────────
# basic-pitch
# ─────────────────────────────────────────────

def test_transcribe_with_fake_model(monkeypatch):
    import basic_pitch.inference as bpi
    calls = []

    def fake_predict(path, model, **kw):
        calls.append(kw)
        print("debug line basic-pitch prints")
        return None, fake_midi([(0.2, 0.7, 60, 80), (0.1, 0.5, 64, 70)]), None
    monkeypatch.setattr(bpi, "predict", fake_predict)
    monkeypatch.setattr(rc, "get_basic_pitch_model", lambda: "model")
    notes = rc.transcribe("x.wav", onset_thresh=0.4)
    assert [x["midi"] for x in notes] == [64, 60]           # sorted by onset
    assert calls[0]["onset_threshold"] == 0.4


def test_basic_pitch_model_loaded_once(monkeypatch):
    import basic_pitch.inference as bpi
    loads = []
    monkeypatch.setattr(bpi, "Model", lambda path: loads.append(path) or object())
    monkeypatch.setattr(rc, "_BP_MODEL", None)
    assert rc.get_basic_pitch_model() is rc.get_basic_pitch_model()
    assert len(loads) == 1


@pytest.mark.slow
def test_real_basic_pitch_on_synthesised_notes(tmp_path, monkeypatch):
    import soundfile as sf
    monkeypatch.setattr(rc, "_BP_MODEL", None)
    wav = tmp_path / "take.wav"
    sf.write(wav, np.concatenate([silence(0.5), tones([60], 1.0, amp=0.3),
                                  silence(0.5), tones([67], 1.0, amp=0.3)]), SR)
    notes = rc.transcribe(wav)
    assert {60, 67} <= {x["midi"] for x in notes}
    c4 = min(x["start"] for x in notes if x["midi"] == 60)
    assert c4 == pytest.approx(0.5, abs=0.1)
