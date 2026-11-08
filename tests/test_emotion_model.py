import json

import numpy as np
import pytest

import emotion_model as em
from conftest import PROFILE, synthetic_rows

@pytest.fixture(autouse=True)
def fewer_trees(monkeypatch):
    monkeypatch.setattr(em, "N_TREES", 50)   # the real model uses 500; 50 keeps tests fast
    monkeypatch.setattr(em, "N_JOBS", 1)     # no worker processes inside the test run


def as_arrays(rows):
    X = np.array([[r[f] for f in em.FEATURES] for r in rows])
    return X, np.array([r["quadrant"] for r in rows]), np.array([r["song"] for r in rows])


def test_evaluate_reports_grouped_cross_validation():
    m = em.evaluate(*as_arrays(synthetic_rows()))
    assert m["accuracy"] > 0.9 and m["arousal_accuracy"] > 0.9
    assert m["classes"] == ["Q1", "Q2", "Q3", "Q4"] and np.sum(m["confusion"]) == 96
    top = sorted(m["importance"], key=m["importance"].get, reverse=True)[:3]
    assert set(top) == {"velocity_mean", "major_minor", "onset_rate"}


def test_folds_never_split_a_song():
    groups = np.array([r["song"] for r in synthetic_rows()])
    folds = em.song_folds(groups)
    assert len(folds) == 5
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
    assert sorted(np.concatenate([t for _, t in folds])) == list(range(len(groups)))


def test_quadrant_profiles():
    profiles = em.quadrant_profiles(synthetic_rows())
    assert profiles["Q1"]["velocity_mean"]["median"] > profiles["Q3"]["velocity_mean"]["median"]
    p = profiles["Q3"]["major_minor"]
    assert p["q25"] <= p["median"] <= p["q75"]


def test_predict_returns_probabilities():
    rows = synthetic_rows()
    X, y, _ = as_arrays(rows)
    bundle = {"model": em.make_forest().fit(X, y), "features": em.FEATURES,
              "classes": ["Q1", "Q2", "Q3", "Q4"]}
    happy = dict(rows[0])
    probs = em.predict(happy, bundle)
    assert max(probs, key=probs.get) == "Q1" and sum(probs.values()) == pytest.approx(1, abs=0.01)


def write_fake_emopia(root):
    """A miniature EMOPIA: label.csv and real MIDI files."""
    import pretty_midi
    root.mkdir(parents=True)
    (root / "midis").mkdir()
    lines = ["ID,4Q,annotator"]
    rng = np.random.default_rng(0)
    for q, (vel, mode, rate) in PROFILE.items():
        for s in range(3):
            clip = f"{q}_song{q}{s}_0"
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(0)
            chord = [60, 64, 67] if mode > 0 else [57, 60, 64]
            step = 1 / rate
            for i in range(int(10 * rate)):
                for p in chord if i % 4 == 0 else [chord[i % 3] + 12]:
                    inst.notes.append(pretty_midi.Note(int(np.clip(rng.normal(vel, 3), 1, 127)),
                                                       p, i * step, i * step + 0.8 * step))
            pm.instruments.append(inst)
            pm.write(str(root / "midis" / f"{clip}.mid"))
            lines.append(f"{clip},{q[1]},X")
    (root / "label.csv").write_text("\n".join(lines) + "\n")
    return root


def test_read_labels_extracts_song_ids(tmp_path):
    root = write_fake_emopia(tmp_path / "EMOPIA_2.2")
    labels = em.read_labels(root)
    assert labels[0] == ("Q1_songQ10_0", "songQ10", "Q1") and len(labels) == 12


def test_build_feature_table_and_cache(tmp_path):
    root = write_fake_emopia(tmp_path / "EMOPIA_2.2")
    cache = tmp_path / "features.csv"
    rows = em.build_feature_table(root, cache)
    assert len(rows) == 12 and cache.exists()
    assert all(r[f] is not None for r in rows for f in em.FEATURES)
    assert em.build_feature_table(root, cache)[0]["velocity_mean"] == pytest.approx(rows[0]["velocity_mean"])


def test_train_end_to_end(tmp_path, monkeypatch):
    rows = synthetic_rows()
    monkeypatch.setattr(em, "download_emopia", lambda: None)
    monkeypatch.setattr(em, "build_feature_table", lambda: rows)
    monkeypatch.setattr(em, "MODEL_PATH", tmp_path / "emotion.joblib")
    monkeypatch.setattr(em, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(em, "REPORT_MD", tmp_path / "report.md")
    monkeypatch.setattr(em, "FIGURE_PATH", tmp_path / "confusion.png")
    bundle = em.train(verbose=False)
    assert (tmp_path / "confusion.png").stat().st_size > 0
    assert "| Random Forest |" in (tmp_path / "report.md").read_text()
    assert em.load(tmp_path / "emotion.joblib")["classes"] == ["Q1", "Q2", "Q3", "Q4"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["metrics"]["accuracy"] == bundle["metrics"]["accuracy"]
    assert "model" not in report


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        em.load(tmp_path / "missing.joblib")


def test_comparison_includes_baselines_on_same_folds():
    X, y, g = as_arrays(synthetic_rows())
    rf = em.evaluate(X, y, g)
    table = em.compare_models(X, y, g, rf)
    assert [r["model"] for r in table] == ["Majority class", "Logistic regression",
                                           "k-NN (k=15)", "SVM (RBF kernel)", "Random Forest"]
    by_name = {r["model"]: r for r in table}
    assert by_name["Majority class"]["accuracy_mean"] < 0.4 < by_name["Random Forest"]["accuracy_mean"]
    assert by_name["Random Forest"]["accuracy_mean"] == rf["accuracy_mean"]
    assert all(r["accuracy_std"] >= 0 for r in table)


def test_markdown_report_tables():
    X, y, g = as_arrays(synthetic_rows())
    metrics = em.evaluate(X, y, g)
    metrics["comparison"] = em.compare_models(X, y, g, metrics)
    md = em.markdown_report({"metrics": metrics, "trained_on": "synthetic"})
    assert md.count("| Majority class |") == 1 and "| **Q3 sad** |" in md
    assert "| velocity_mean |" in md
