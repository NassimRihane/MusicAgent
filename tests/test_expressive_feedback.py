import json

import numpy as np
import pytest

import emotion_model as em
import expressive_feedback as xf
from conftest import synthetic_rows, write_json


@pytest.fixture(scope="module")
def bundle():
    """An emotion model trained on synthetic features (see conftest.synthetic_rows)."""
    rows = synthetic_rows()
    X = np.array([[r[f] for f in em.FEATURES] for r in rows])
    y = np.array([r["quadrant"] for r in rows])
    model = em.make_forest().set_params(n_estimators=50, n_jobs=1).fit(X, y)
    importance = {f: 0.01 for f in em.FEATURES}
    importance.update(velocity_mean=0.3, articulation=0.05, timing_variability=0.02)
    return {"model": model, "features": em.FEATURES, "classes": list(model.classes_),
            "profiles": em.quadrant_profiles(rows), "trained_on": "synthetic",
            "metrics": {"importance": importance}}


def loud_major(bundle):
    """Features of a typical Q1 (happy) clip."""
    return {f: bundle["profiles"]["Q1"][f]["median"] for f in em.FEATURES}


@pytest.mark.parametrize("word,quadrant", [("sad", "Q3"), ("Happy", "Q1"), ("calm", "Q4"),
                                           ("tense", "Q2"), ("q3", "Q3")])
def test_resolve_target(word, quadrant):
    assert xf.resolve_target(word) == quadrant


def test_resolve_unknown_emotion():
    with pytest.raises(ValueError):
        xf.resolve_target("purple")


def test_gap_in_interquartile_ranges():
    assert xf.gap(70, {"median": 50, "q25": 40, "q75": 60}) == 1.0
    assert xf.gap(45, {"median": 50, "q25": 40, "q75": 60}) == -0.25
    assert np.isfinite(xf.gap(1, {"median": 1, "q25": 1, "q75": 1}))


def test_happy_performance_aiming_for_sad(bundle):
    result = xf.analyse(loud_major(bundle), "Q3", bundle)
    assert result["heard_as"] == "Q1" and result["target"] == "Q3"
    first = result["advice"][0]
    assert first["cue"] == "velocity_mean" and first["advice"].startswith("Play softer")
    w = result["what_if"]["target_probability"]
    assert w["after"] > w["before"]


def test_advice_only_on_controllable_cues(bundle):
    result = xf.analyse(loud_major(bundle), "Q3", bundle)
    assert {c["cue"] for c in result["advice"]} <= set(xf.ACTIONABLE)
    assert {c["cue"] for c in result["fixed_by_piece"]} == set(xf.FIXED_BY_PIECE)
    assert len(result["advice"]) <= xf.MAX_ADVICE


def test_no_advice_when_already_there(bundle):
    target = {f: bundle["profiles"]["Q3"][f]["median"] for f in em.FEATURES}
    result = xf.analyse(target, "Q3", bundle)
    assert result["advice"] == [] and len(result["already_there"]) == len(xf.ACTIONABLE)
    w = result["what_if"]["target_probability"]
    assert w["before"] == w["after"]


def test_priority_combines_gap_and_importance(bundle):
    result = xf.analyse(loud_major(bundle), "Q3", bundle)
    priorities = [c["priority"] for c in result["advice"]]
    assert priorities == sorted(priorities, reverse=True)
    c = result["advice"][0]
    assert c["priority"] == pytest.approx(abs(c["gap"]) * c["importance"], rel=0.01)   # gap is rounded


def test_main_writes_expressive_json(tmp_path, bundle, monkeypatch):
    monkeypatch.setattr(em, "load", lambda *a, **k: bundle)
    notes = [{"start": 0.3 * i, "end": 0.3 * i + 0.1, "midi": [60, 64, 67][i % 3],
              "velocity": 95} for i in range(30)]
    session = write_json(tmp_path / "session_x.json", {"notes": notes})
    xf.main([str(session), "--emotion", "sad"])
    out = json.loads((tmp_path / "expressive_session_x.json").read_text())
    assert out["emotion"] == "sad" and out["target"] == "Q3"
    # synthetic profiles have unrealistic tempo values, so only check that loudness is advised
    assert any(c["advice"].startswith("Play softer") for c in out["advice"])


def test_main_needs_notes(tmp_path, bundle, monkeypatch):
    monkeypatch.setattr(em, "load", lambda *a, **k: bundle)
    session = write_json(tmp_path / "session_x.json", {"notes": []})
    with pytest.raises(SystemExit):
        xf.main([str(session), "--emotion", "sad"])
