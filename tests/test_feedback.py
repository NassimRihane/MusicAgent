import json

import numpy as np
import pytest

import feedback as fb
import recognition as rc
from conftest import write_json

C_MAJOR = rc.scale_pitch_classes("C major")


def n(start, end, midi, velocity=80):
    return {"start": start, "end": end, "midi": midi, "velocity": velocity}


def steady_notes(period=0.5, count=32, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(count) * period + rng.normal(0, jitter, count)
    return [n(float(t), float(t) + 0.3, [60, 64, 67, 72][i % 4]) for i, t in enumerate(times)]


# ─────────────────────────────────────────────
# Level 1 — notes
# ─────────────────────────────────────────────

def test_notes_in_and_out_of_key():
    res = fb.analyse_notes([n(0, 1, 60), n(1, 2, 61), n(2, 3, 61), n(3, 4, 64)], C_MAJOR)
    assert (res["total_notes"], res["in_scale_count"], res["score"]) == (4, 2, 50)
    assert res["frequent_errors"][0] == {"note": "C#", "count": 2}
    assert res["assessment"].startswith("Many notes") and res["avg_velocity"] == 80


def test_raised_seventh_is_in_a_minor():
    res = fb.analyse_notes([n(0, 1, 68), n(1, 2, 69)], rc.scale_pitch_classes("A minor"))  # G#, A
    assert res["score"] == 100


def test_notes_no_data():
    assert fb.analyse_notes([], C_MAJOR)["status"] == "no_data"


# ─────────────────────────────────────────────
# Level 2 — rhythm
# ─────────────────────────────────────────────

def rhythm(notes, target=None):
    return fb.analyse_rhythm(notes, rc.track_beats(notes=notes), target)


def test_rhythm_steady():
    res = rhythm(steady_notes())
    assert res["timing_assessment"] == "Steady tempo" and res["timing_score"] >= 90
    assert res["tempo_bpm"] == pytest.approx(120, rel=0.05)


def tempo_curve(slowest_period, count=40):
    """Ritardando from 120 BPM down to 60/slowest_period, then accelerando back."""
    periods = np.concatenate([np.linspace(0.5, slowest_period, count // 2),
                              np.linspace(slowest_period, 0.5, count // 2)])
    times = np.concatenate([[0], np.cumsum(periods)[:-1]])
    return [n(float(t), float(t) + 0.3, 60) for t in times]


def test_rhythm_tempo_changes_score_lower():
    # Short random jitter is measured unreliably (see CLAUDE.md); real tempo changes are not
    scores = [rhythm(tempo_curve(p))["timing_score"] for p in (0.5, 0.6, 0.9)]
    assert scores[0] > scores[1] > scores[2]
    assert "varies" in rhythm(tempo_curve(0.9))["timing_assessment"]


def test_rhythm_tempo_against_target():
    res = rhythm(steady_notes(), target=100)
    assert res["tempo_vs_target"] == pytest.approx(0.2, abs=0.03)


def test_rhythm_hesitation():
    notes = steady_notes(count=16) + [n(t, t + 0.3, 60) for t in np.arange(12.0, 16.0, 0.5)]
    res = rhythm(notes)
    assert res["hesitation_count"] == 1 and res["hesitations"][0]["gap_sec"] == pytest.approx(4.5)


def test_rhythm_not_enough_notes():
    res = rhythm([n(0, 0.3, 60), n(0.5, 0.8, 62)])
    assert "timing_score" not in res and "Not enough" in res["timing_assessment"]


def test_rhythm_no_data():
    assert fb.analyse_rhythm([], {"tempo_bpm": 0, "beat_times": [], "tempo_cv": 0})["status"] == "no_data"


# ─────────────────────────────────────────────
# Level 3 — harmony
# ─────────────────────────────────────────────

def chord(name, numeral, fit="in_scale"):
    return {"chord": name, "numeral": numeral, "fit": fit}


def test_harmony_ignores_silence_and_finds_progressions():
    chords = [chord("N", None, "no_chord"), chord("C", "I"), chord("C/E", "I"), chord("F", "IV"),
              chord("G7", "V"), chord("F#", "#IV", "out_of_scale")]
    res = fb.analyse_harmony(chords)
    assert res["total_chord_segments"] == 5 and res["harmony_score"] == 80
    assert res["progression"] == "I IV V #IV"
    assert res["progressions_detected"][0]["progression"] == "I-IV-V"
    assert res["non_diatonic_chords"] == [{"chord": "F#", "count": 1}]


def test_harmony_minor_progression():
    res = fb.analyse_harmony([chord("Am", "i"), chord("Dm", "iv"), chord("E", "V")])
    assert res["progressions_detected"][0]["progression"] == "i-iv-V"


def test_harmony_no_chords():
    assert fb.analyse_harmony([chord("N", None, "no_chord")])["status"] == "no_data"


# ─────────────────────────────────────────────
# Suggestions
# ─────────────────────────────────────────────

def test_suggestions_empty_session():
    sugg = fb.build_suggestions({"status": "no_data"}, {}, {})
    assert len(sugg) == 1 and "No notes" in sugg[0]["message"]


def test_suggestions_low_note_accuracy():
    notes = {"score": 40, "frequent_errors": [{"note": "C#", "count": 3}]}
    sugg = fb.build_suggestions(notes, {"timing_score": 100}, {})
    assert sugg[0]["priority"] == 1 and "C#." in sugg[0]["message"]


def test_suggestions_metronome_mentions_rubato_and_target_tempo():
    sugg = fb.build_suggestions({"score": 95}, {"timing_score": 20, "target_bpm": 100,
                                                "tempo_bpm": 90}, {})
    msg = next(s["message"] for s in sugg if s["area"] == "rhythm")
    assert "rubato" in msg and "70 BPM" in msg


def test_suggestions_encouragement_needs_a_timing_score():
    assert not any(s["area"] == "encouragement" for s in fb.build_suggestions({"score": 95}, {}, {}))
    good = fb.build_suggestions({"score": 95}, {"timing_score": 90}, {})
    assert any(s["area"] == "encouragement" for s in good)


# ─────────────────────────────────────────────
# End to end
# ─────────────────────────────────────────────

def test_main_writes_reports_and_auto_detects_chords(tmp_path):
    notes = steady_notes()
    session = write_json(tmp_path / "session_t.json",
                         {"meta": {"scale": "C major", "scale_source": "declared"}, "notes": notes})
    write_json(tmp_path / "session_t_chords.json",
               {"chords": [chord("C", "I"), chord("G", "V"), chord("C", "I")]})
    fb.main([str(session)])
    out = json.loads((tmp_path / "feedback_session_t.json").read_text())
    assert out["note_accuracy"]["score"] == 100
    assert out["rhythm"]["timing_assessment"] == "Steady tempo"
    assert out["harmony"]["harmony_score"] == 100
    assert "Assessment  : Excellent" in (tmp_path / "feedback_session_t.txt").read_text()


def test_main_detects_key_for_sessions_without_one(tmp_path):
    session = write_json(tmp_path / "session_t.json", {"meta": {}, "notes": steady_notes()})
    fb.main([str(session)])
    assert json.loads((tmp_path / "feedback_session_t.json").read_text())["meta"]["scale"] == "C major"
