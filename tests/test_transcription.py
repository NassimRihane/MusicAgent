import json

import numpy as np
import pytest
import soundfile as sf

import recognition as rc
import transcription as tr
from conftest import SR, silence, tones

# A C major arpeggio at 120 BPM
NOTES = [{"start": 0.5 * i, "end": 0.5 * i + 0.4, "midi": m, "velocity": 80}
         for i, m in enumerate([60, 64, 67, 72] * 4)]


@pytest.fixture
def fake_basic_pitch(monkeypatch):
    calls = []

    def fake(path, *args, **kw):
        calls.append(str(path))
        return [dict(n) for n in NOTES]
    monkeypatch.setattr(rc, "transcribe", fake)
    return calls


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "session_test.wav"
    audio = np.concatenate([np.concatenate([tones([n["midi"]], 0.4), silence(0.1)]) for n in NOTES])
    sf.write(path, audio, SR)
    return path


def test_build_session_detects_key_and_tempo(wav, fake_basic_pitch):
    out = tr.build_session(wav, source="test")
    session = json.loads(out.read_text())
    meta = session["meta"]
    assert out == wav.with_suffix(".json") and fake_basic_pitch == [str(wav)]
    assert meta["scale"] == meta["key_detected"] == "C major"
    assert meta["scale_source"].startswith("detected") and meta["bpm_source"].startswith("estimated")
    assert meta["bpm"] == pytest.approx(120, rel=0.05)
    assert len(session["notes"]) == 16 and len(session["events"]) == 16
    assert all(e["in_scale"] for e in session["events"])


def test_build_session_uses_declared_key_and_tempo(wav, fake_basic_pitch):
    meta = json.loads(tr.build_session(wav, "G major", 90).read_text())["meta"]
    assert (meta["scale"], meta["bpm"]) == ("G major", 90)
    assert meta["scale_source"] == meta["bpm_source"] == "declared"
    assert meta["key_detected"] == "C major"


def test_import_audio_converts_to_mono_22050(tmp_path):
    src = tmp_path / "stereo.wav"
    sf.write(src, np.stack([tones([60], 1.0, sr=44100)] * 2, axis=1), 44100)
    out = tr.import_audio(src, tmp_path / "session_x.wav")
    audio, sr = sf.read(out)
    assert sr == SR and audio.ndim == 1 and len(audio) == pytest.approx(SR, abs=10)


def test_live_preview_segment_times(monkeypatch):
    monkeypatch.setattr(rc, "transcribe", lambda path: [
        {"start": 0.1, "end": 0.6, "midi": 60, "velocity": 80},
        {"start": 1.1, "end": 1.5, "midi": 61, "velocity": 60}])
    events = tr.analyse_segment(np.zeros(SR, dtype=np.float32), 20.0,
                                rc.scale_pitch_classes("C major"), 60)
    assert [e["time"] for e in events] == [20.1, 21.1]
    assert [e["in_scale"] for e in events] == [True, False]


def test_display_event(capsys):
    tr.display_event(rc.notes_to_events(NOTES[:1])[0])
    assert "C4" in capsys.readouterr().out


def test_main_with_audio_file(tmp_path, wav, fake_basic_pitch, monkeypatch):
    monkeypatch.setattr(tr, "SESSIONS_DIR", tmp_path / "sessions")
    tr.main(["--audio", str(wav)])
    [session] = (tmp_path / "sessions").glob("session_*.json")
    meta = json.loads(session.read_text())["meta"]
    assert meta["source"] == wav.name and session.with_suffix(".wav").exists()
