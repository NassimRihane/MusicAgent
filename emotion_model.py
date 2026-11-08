# emotion_model.py
#
# Layer 3 — Random Forest emotion recogniser
#
# Predicts which emotion a piano performance conveys, using a scikit-learn
# RandomForestClassifier on the note-level descriptors of
# expression_analysis.NOTE_FEATURES (tempo, articulation, dynamics, ...).
#
# Training data: EMOPIA (Hung et al., ISMIR 2021; CC-BY 4.0) — 1,078 pop
# piano clips, transcribed to MIDI, each labelled with a quadrant of
# Russell's valence–arousal plane:
#
#   Q1  positive valence, high arousal   happy / excited
#   Q2  negative valence, high arousal   tense / angry
#   Q3  negative valence, low arousal    sad
#   Q4  positive valence, low arousal    calm / relaxed
#
# Evaluation is 5-fold cross-validation grouped by song, so clips of the same
# song are never split between training and test. Feature importance is
# measured by permutation importance on the test folds. The Random Forest is
# compared with standard classifiers (majority class, logistic regression,
# k-NN, SVM) on the same folds.
#
# Training writes, in models/:
#   emotion_rf.joblib          the model, feature profiles and metrics
#   emotion_rf_report.json     metrics, comparison table, importances
#   emotion_rf_report.md       the same as tables, ready for a paper
#   emotion_confusion.png      confusion matrix figure
#
# Usage:
#   python emotion_model.py --train                 # downloads EMOPIA if needed (32 MB)
#   python emotion_model.py --predict sessions/session_X.json
#   python emotion_model.py --predict take.mp3


import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np

import expression_analysis as ea
import recognition as rc



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Random Forest emotion recogniser (EMOPIA)")
    parser.add_argument("--train", action="store_true", help="Train and evaluate the model")
    parser.add_argument("--predict", default=None, help="Session JSON or audio file to classify")
    return parser.parse_args(argv)



# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
EMOPIA_DIR  = DATA_DIR / "EMOPIA_2.2"
EMOPIA_URL  = "https://zenodo.org/records/5257995/files/EMOPIA_2.2.zip?download=1"
FEATURES_CSV = DATA_DIR / "emopia_features.csv"
MODEL_PATH  = ROOT / "models" / "emotion_rf.joblib"
REPORT_PATH = ROOT / "models" / "emotion_rf_report.json"
REPORT_MD   = ROOT / "models" / "emotion_rf_report.md"
FIGURE_PATH = ROOT / "models" / "emotion_confusion.png"

FEATURES = ea.NOTE_FEATURES
N_TREES  = 500
N_JOBS   = -1    # all CPU cores

QUADRANTS = {
    "Q1": "happy / excited",
    "Q2": "tense / angry",
    "Q3": "sad",
    "Q4": "calm / relaxed",
}

# Words a learner may use for their intention → quadrant
EMOTION_TO_QUADRANT = {
    "happy": "Q1", "joyful": "Q1", "excited": "Q1", "cheerful": "Q1",
    "tense": "Q2", "angry": "Q2", "anxious": "Q2", "dramatic": "Q2",
    "sad": "Q3", "melancholic": "Q3", "depressed": "Q3",
    "calm": "Q4", "relaxed": "Q4", "peaceful": "Q4", "tender": "Q4",
}



# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────

def download_emopia(data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip EMOPIA 2.2 from Zenodo (32 MB) if it isn't there yet."""
    import urllib.request
    target = data_dir / "EMOPIA_2.2"
    if (target / "label.csv").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "EMOPIA_2.2.zip"
    print("  Downloading EMOPIA (32 MB) from Zenodo...")
    urllib.request.urlretrieve(EMOPIA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    return target


def read_labels(emopia_dir: Path = EMOPIA_DIR) -> list:
    """[(clip_id, song_id, quadrant)] from label.csv."""
    rows = []
    with open(emopia_dir / "label.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            clip = r["ID"]
            song = clip[3:clip.rindex("_")]   # 'Q1_0vLPYiPN7qY_2' → '0vLPYiPN7qY'
            rows.append((clip, song, f"Q{r['4Q']}"))
    return rows


def build_feature_table(emopia_dir: Path = EMOPIA_DIR, cache: Path = FEATURES_CSV) -> list:
    """Note features of every labelled clip (cached in a CSV after the first run)."""
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return [{k: (float(v) if k in FEATURES else v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
    rows, t0 = [], time.time()
    labels = read_labels(emopia_dir)
    for i, (clip, song, quadrant) in enumerate(labels):
        midi = emopia_dir / "midis" / f"{clip}.mid"
        if not midi.exists():
            continue
        feats = ea.note_features(rc.load_midi_notes(midi, sustain=True))
        if any(v is None for v in feats.values()):
            continue
        rows.append({"clip": clip, "song": song, "quadrant": quadrant, **feats})
        if (i + 1) % 200 == 0:
            print(f"  features: {i + 1}/{len(labels)} clips ({time.time() - t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "song", "quadrant"] + FEATURES)
        w.writeheader()
        w.writerows(rows)
    return rows



# ─────────────────────────────────────────────
# 3. TRAINING AND EVALUATION
# ─────────────────────────────────────────────

def make_forest(seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                                  class_weight="balanced", random_state=seed, n_jobs=N_JOBS)


def song_folds(groups: np.ndarray, n_splits: int = 5) -> list:
    """(train, test) index pairs where each song is entirely in one test fold."""
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def arousal(q: str) -> str:
    return "high" if q in ("Q1", "Q2") else "low"


def valence(q: str) -> str:
    return "positive" if q in ("Q1", "Q4") else "negative"


def scores(y: np.ndarray, pred: np.ndarray, fold_accuracies: list) -> dict:
    """Accuracy (overall, mean ± std over folds), macro-F1, arousal and valence accuracy."""
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy":         round(float(accuracy_score(y, pred)), 3),
        "accuracy_mean":    round(float(np.mean(fold_accuracies)), 3),
        "accuracy_std":     round(float(np.std(fold_accuracies)), 3),
        "macro_f1":         round(float(f1_score(y, pred, average="macro")), 3),
        "arousal_accuracy": round(float(np.mean([arousal(a) == arousal(b) for a, b in zip(y, pred)])), 3),
        "valence_accuracy": round(float(np.mean([valence(a) == valence(b) for a, b in zip(y, pred)])), 3),
    }


def cross_validate(model_factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Song-grouped predictions for every clip, and the accuracy of each fold."""
    pred, fold_accuracies, models = np.empty_like(y), [], []
    for train, test in song_folds(groups):
        model = model_factory().fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        fold_accuracies.append(float(np.mean(pred[test] == y[test])))
        models.append((model, test))
    return pred, fold_accuracies, models


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """5-fold cross-validation grouped by song + permutation importance on the test folds."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import confusion_matrix

    classes = sorted(set(y))
    pred, fold_accuracies, models = cross_validate(lambda: make_forest(seed), X, y, groups)
    importances = [permutation_importance(m, X[test], y[test], n_repeats=10,
                                          random_state=seed).importances_mean
                   for m, test in models]
    return {
        **scores(y, pred, fold_accuracies),
        "chance":     round(max(np.mean(y == c) for c in classes), 3),
        "classes":    classes,
        "confusion":  confusion_matrix(y, pred, labels=classes).tolist(),
        "importance": {f: round(float(v), 4) for f, v in zip(FEATURES, np.mean(importances, axis=0))},
    }


def baseline_models(seed: int = 0) -> dict:
    """Standard classifiers the Random Forest is compared with (features standardised)."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return {
        "Majority class":      lambda: DummyClassifier(strategy="most_frequent"),
        "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                                   max_iter=2000, class_weight="balanced")),
        "k-NN (k=15)":         lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "SVM (RBF kernel)":    lambda: make_pipeline(StandardScaler(), SVC(
                                   class_weight="balanced", random_state=seed)),
    }


def compare_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, rf_metrics: dict,
                   seed: int = 0) -> list:
    """Comparison table: each baseline on the same folds, then the Random Forest."""
    rows = []
    for name, factory in baseline_models(seed).items():
        pred, fold_accuracies, _ = cross_validate(factory, X, y, groups)
        rows.append({"model": name, **scores(y, pred, fold_accuracies)})
    keys = ("accuracy", "accuracy_mean", "accuracy_std", "macro_f1",
            "arousal_accuracy", "valence_accuracy")
    rows.append({"model": "Random Forest", **{k: rf_metrics[k] for k in keys}})
    return rows


def quadrant_profiles(rows: list) -> dict:
    """Median and inter-quartile range of each feature per quadrant."""
    profiles = {}
    for q in QUADRANTS:
        values = np.array([[r[f] for f in FEATURES] for r in rows if r["quadrant"] == q])
        q25, med, q75 = np.percentile(values, [25, 50, 75], axis=0)
        profiles[q] = {f: {"median": round(float(m), 3), "q25": round(float(a), 3),
                           "q75": round(float(b), 3)}
                       for f, a, m, b in zip(FEATURES, q25, med, q75)}
    return profiles


def train(seed: int = 0, verbose: bool = True) -> dict:
    """Download EMOPIA if needed, evaluate by cross-validation, fit on all clips, save."""
    import joblib
    download_emopia()
    rows = build_feature_table()
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    y = np.array([r["quadrant"] for r in rows])
    groups = np.array([r["song"] for r in rows])

    metrics = evaluate(X, y, groups, seed)
    metrics["comparison"] = compare_models(X, y, groups, metrics, seed)
    model = make_forest(seed).fit(X, y)
    bundle = {"model": model, "features": FEATURES, "classes": list(model.classes_),
              "profiles": quadrant_profiles(rows), "metrics": metrics,
              "trained_on": f"EMOPIA 2.2 ({len(rows)} clips, {len(set(groups))} songs)"}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps({k: v for k, v in bundle.items() if k != "model"},
                                      indent=2), encoding="utf-8")
    REPORT_MD.write_text(markdown_report(bundle), encoding="utf-8")
    save_confusion_figure(metrics, FIGURE_PATH)
    if verbose:
        print_report(bundle)
    return bundle


def markdown_report(bundle: dict) -> str:
    """The evaluation as Markdown tables (comparison, confusion matrix, importance)."""
    m = bundle["metrics"]
    lines = [f"# Emotion recognition — {bundle['trained_on']}", "",
             "5-fold cross-validation grouped by song (no song in both training and test).",
             "Accuracy: mean ± standard deviation over the 5 folds.", "",
             "| Model | 4 quadrants | Macro-F1 | Arousal | Valence |",
             "|---|---|---|---|---|"]
    for r in m["comparison"]:
        lines.append(f"| {r['model']} | {r['accuracy_mean']:.1%} ± {r['accuracy_std']:.1%} | "
                     f"{r['macro_f1']:.2f} | {r['arousal_accuracy']:.1%} | {r['valence_accuracy']:.1%} |")
    lines += ["", "## Confusion matrix (Random Forest; rows = true, columns = predicted)", "",
              "| | " + " | ".join(f"{c} {QUADRANTS[c]}" for c in m["classes"]) + " |",
              "|---" * (len(m["classes"]) + 1) + "|"]
    for c, row in zip(m["classes"], m["confusion"]):
        lines.append(f"| **{c} {QUADRANTS[c]}** | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Permutation importance (mean accuracy drop when a feature is shuffled)", "",
              "| Feature | Importance |", "|---|---|"]
    for f, v in sorted(m["importance"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {f} | {v:+.4f} |")
    return "\n".join(lines) + "\n"


def save_confusion_figure(metrics: dict, path: Path):
    """Row-normalised confusion matrix as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(metrics["confusion"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    labels = [f"{c}\n{QUADRANTS[c]}" for c in metrics["classes"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.5 else "#14171c")
    ax.set_xticks(range(len(cm)), labels, fontsize=8)
    ax.set_yticks(range(len(cm)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Random Forest — accuracy {metrics['accuracy']:.1%}", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_report(bundle: dict):
    m = bundle["metrics"]
    print(f"\n  Emotion Random Forest — {bundle['trained_on']}")
    print(f"  4-quadrant accuracy : {m['accuracy_mean']:.1%} ± {m['accuracy_std']:.1%} over folds "
          f"(chance {m['chance']:.1%}, macro-F1 {m['macro_f1']:.2f})")
    print(f"  Arousal (high/low)  : {m['arousal_accuracy']:.1%}")
    print(f"  Valence (pos/neg)   : {m['valence_accuracy']:.1%}")
    print("  Confusion (rows = true, cols = predicted):")
    print("        " + "  ".join(f"{c:>4}" for c in m["classes"]))
    for c, row in zip(m["classes"], m["confusion"]):
        print(f"    {c}  " + "  ".join(f"{v:>4}" for v in row))
    print("  Permutation importance:")
    for f, v in sorted(m["importance"].items(), key=lambda kv: -kv[1]):
        print(f"    {f:<20} {v:+.4f}")
    if m.get("comparison"):
        print("  Comparison (same folds):")
        print(f"    {'model':<22} {'4 quadrants':>15} {'F1':>5} {'arousal':>8} {'valence':>8}")
        for r in m["comparison"]:
            print(f"    {r['model']:<22} {r['accuracy_mean']:6.1%} ± {r['accuracy_std']:4.1%} "
                  f"{r['macro_f1']:5.2f} {r['arousal_accuracy']:8.1%} {r['valence_accuracy']:8.1%}")
    print(f"  Saved: {MODEL_PATH}")
    print(f"         {REPORT_MD}  |  {FIGURE_PATH}")



# ─────────────────────────────────────────────
# 4. PREDICTION
# ─────────────────────────────────────────────

def load(path: Path = MODEL_PATH) -> dict:
    import joblib
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Train it first: "
                                "python emotion_model.py --train")
    return joblib.load(path)


def predict(features: dict, bundle: dict = None) -> dict:
    """{quadrant: probability} for one performance's note features."""
    bundle = bundle or load()
    x = np.array([[features[f] for f in bundle["features"]]])
    probs = bundle["model"].predict_proba(x)[0]
    return {q: round(float(p), 3) for q, p in zip(bundle["classes"], probs)}



# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    if args.train:
        train()
    if args.predict:
        notes, _ = ea.load_input(args.predict)
        probs = predict(ea.note_features(notes))
        print(f"\n  {Path(args.predict).name}")
        for q, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            print(f"    {q} {QUADRANTS[q]:<16} {p:.0%}")
    if not (args.train or args.predict):
        print("Nothing to do. Use --train and/or --predict FILE.")


if __name__ == "__main__":
    main()
