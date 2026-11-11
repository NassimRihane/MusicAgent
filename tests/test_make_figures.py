import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import make_figures as mf
from conftest import write_json

QUADS = ["Q1", "Q2", "Q3", "Q4"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def metric_row(model, acc):
    return {"model": model, "accuracy": acc, "accuracy_mean": acc, "accuracy_std": 0.02,
            "macro_f1": acc, "arousal_accuracy": 0.8, "valence_accuracy": 0.7}


def write_reports(models, sessions):
    write_json(models / "emotion_rf_report.json", {
        "trained_on": "test data",
        "metrics": {"accuracy": 0.6, "classes": QUADS,
                    "confusion": [[5, 1, 0, 0], [1, 5, 0, 0], [0, 0, 5, 1], [0, 0, 1, 5]],
                    "importance": {"velocity_mean": 0.2, "tempo_bpm": -0.001},
                    "comparison": [metric_row("Majority class", 0.3),
                                   metric_row("Random Forest", 0.6)]}})
    summary = {b: {"all": {m: {"mean": 0.8, "std": 0.1} for m in mf.METRICS}}
               for b in ("rf", "music21")}
    write_json(models / "chord_eval_report.json", {"pieces": 3, "summary": summary})
    write_json(models / "chord_eval_recording_take.json", {
        "scores": {"rf": {"root": 1.0, "majmin": 1.0, "sevenths": 0.5},
                   "music21": {"root": 0.9, "majmin": 1.0, "sevenths": 1.0}},
        "chords": [{"number": 1, "expected": "C:maj", "rf": "C:maj", "music21": "C:maj"},
                   {"number": 2, "expected": "G:7", "rf": "G:maj", "music21": "G:7"}]})
    cue = lambda name, gap, prio: {"cue": name, "name": name.title(), "gap": gap,
                                   "priority": prio, "advice": f"Change {name}"}
    write_json(sessions / "expressive_session_x.json", {
        "input": "session_x.json", "emotion": "sad", "target": "Q3", "target_name": "sad",
        "heard_as": "Q1", "heard_as_name": "happy / excited",
        "probabilities": {"Q1": 0.5, "Q2": 0.1, "Q3": 0.2, "Q4": 0.2},
        "advice": [cue("velocity_mean", 1.2, 0.3)], "already_there": [cue("tempo_bpm", 0.1, 0.0)],
        "what_if": {"target_probability": {"before": 0.2, "after": 0.35}}})


def test_all_figures_and_captions(tmp_path, monkeypatch):
    models, sessions, out = tmp_path / "models", tmp_path / "sessions", tmp_path / "figures"
    models.mkdir(); sessions.mkdir()
    write_reports(models, sessions)
    monkeypatch.setattr(mf, "MODELS", models)
    monkeypatch.setattr(mf, "ROOT", tmp_path)
    mf.main(["--out", str(out), "--expression", str(tmp_path / "missing.json")])

    expected = ["emotion_model_comparison", "emotion_confusion", "emotion_feature_importance",
                "chord_eval_comparison", "chord_eval_by_quality", "advice_example"]
    for name in expected:
        assert (out / f"{name}.png").stat().st_size > 0, name
        assert (out / f"{name}.pdf").stat().st_size > 0, name
    captions = (out / "captions.md").read_text()
    assert all(f"## {name}" in captions for name in expected)
    assert "raise the intended emotion from 20% to 35%" in captions
    assert "recognises 0 of 1 seventh chords" in captions


def test_missing_reports_are_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mf, "MODELS", tmp_path)
    monkeypatch.setattr(mf, "ROOT", tmp_path)
    mf.main(["--out", str(tmp_path / "figures"), "--expression", str(tmp_path / "none.json")])
    out = capsys.readouterr().out
    assert out.count("(skipped") == 4
    assert (tmp_path / "figures" / "captions.md").exists()
