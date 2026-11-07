import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import soundfile as sf

import recognition as rc
import piano_roll as pr
from conftest import SR, event, make_session, silence, write_json


def n(start, end, midi, velocity=80):
    return {"start": start, "end": end, "midi": midi, "velocity": velocity}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_load_session_json(tmp_path):
    path = write_json(tmp_path / "session_x.json",
                      make_session([event(0, [60]), event(1, [62])]))
    [rec] = pr.load_recordings(path)
    assert rec["label"] == "session_x" and len(rec["notes"]) == 2


def test_load_expression_json_gives_both_recordings(tmp_path):
    rec = lambda f: {"file": f, "cues": {"duration_sec": 3.0}, "notes": [n(0, 1, 60)]}
    path = write_json(tmp_path / "expression.json",
                      {"recordings": [rec("happy.mp3"), rec("sad.mp3")]})
    assert [r["label"] for r in pr.load_recordings(path)] == ["happy", "sad"]


def test_load_audio_transcribes(tmp_path, monkeypatch):
    wav = tmp_path / "take.wav"
    sf.write(wav, silence(2.0), SR)
    monkeypatch.setattr(rc, "transcribe", lambda path: [n(0.5, 1.0, 60)])
    [rec] = pr.load_recordings(wav)
    assert rec["duration"] == pytest.approx(2.0) and rec["notes"][0]["midi"] == 60


def test_load_rejects_unknown_json(tmp_path):
    with pytest.raises(ValueError):
        pr.load_recordings(write_json(tmp_path / "x.json", {"foo": 1}))


def test_plot_shares_axes_and_describes_recordings():
    recs = [{"label": "Happy", "notes": [n(0, 1, 60), n(1, 2, 64), n(2, 3, 67)], "duration": 3.0},
            {"label": "Sad", "notes": [n(0, 2, 57), n(2, 4, 60), n(4, 6, 64)], "duration": None}]
    fig = pr.plot_piano_rolls(recs, title="Test")
    axes = fig.axes
    assert len(axes) == 2
    assert axes[0].get_ylim() == axes[1].get_ylim() == (48, 72)
    assert axes[0].get_xlim() == (0, 6)                     # longest recording
    texts = [t.get_text() for ax in axes for t in ax.texts]
    assert "C major  ·  3.0 s  ·  3 notes" in texts
    assert "A minor  ·  6.0 s  ·  3 notes" in texts


def test_plot_handles_empty_recording():
    fig = pr.plot_piano_rolls([{"label": "Silence", "notes": [], "duration": 1.0}])
    assert "no notes detected" in [t.get_text() for t in fig.axes[0].texts]


def test_too_many_recordings():
    recs = [{"label": str(i), "notes": [], "duration": 1.0} for i in range(len(pr.PALETTE) + 1)]
    with pytest.raises(ValueError):
        pr.plot_piano_rolls(recs)


def test_main_writes_image_with_labels(tmp_path):
    path = write_json(tmp_path / "session_x.json",
                      make_session([event(0, [60]), event(1, [64])]))
    out = tmp_path / "roll.png"
    pr.main([str(path), "--labels", "Mine", "--out", str(out)])
    assert out.stat().st_size > 0


def test_main_label_count_mismatch(tmp_path):
    path = write_json(tmp_path / "session_x.json", make_session([event(0, [60])]))
    with pytest.raises(SystemExit):
        pr.main([str(path), "--labels", "A", "B"])
