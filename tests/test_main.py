import json
import os
import time

import numpy as np
import pytest
import soundfile as sf

import main as pipeline
from conftest import SR, event, make_session, silence, tones, write_json


def test_newest_session_ignores_chord_files_and_old_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSIONS_DIR", tmp_path)
    old = write_json(tmp_path / "session_old.json", {})
    os.utime(old, (1_000, 1_000))
    new = write_json(tmp_path / "session_new.json", {})
    write_json(tmp_path / "session_new_chords.json", {})
    assert pipeline.newest_session(since=time.time() - 60) == new
    assert pipeline.newest_session(since=time.time() + 60) is None


def test_run_step_reports_failure(tmp_path):
    ok, bad = tmp_path / "ok.py", tmp_path / "bad.py"
    ok.write_text("pass")
    bad.write_text("raise SystemExit(3)")
    assert pipeline.run_step("ok", str(ok)) is True
    assert pipeline.run_step("bad", str(bad)) is False


def test_missing_session_exits(tmp_path):
    with pytest.raises(SystemExit):
        pipeline.main(["--session", str(tmp_path / "nope.json")])


@pytest.mark.slow
def test_full_pipeline_dry_run(tmp_path, capfd):
    """chord analysis → score → feedback → agent (--dry-run), as subprocesses."""
    audio = np.concatenate([silence(0.5), tones([60, 64, 67], 2.0), tones([55, 59, 62], 2.0)])
    sf.write(tmp_path / "session_t.wav", audio, SR)
    session = write_json(tmp_path / "session_t.json",
                         make_session([event(0.5 + i * 0.5, [[60, 64, 67, 72][i % 4]])
                                       for i in range(8)]))

    pipeline.main(["--session", str(session), "--score", "--dry-run"])

    for name in ["session_t_chords.json", "session_t.musicxml",
                 "feedback_session_t.json", "feedback_session_t.txt"]:
        assert (tmp_path / name).exists(), name
    fb = json.loads((tmp_path / "feedback_session_t.json").read_text())
    assert fb["harmony"]["harmony_score"] == 100
    out = capfd.readouterr().out
    assert "--- SESSION DATA ---" in out
    assert not (tmp_path / "conversation_feedback_session_t.json").exists()   # no chat
