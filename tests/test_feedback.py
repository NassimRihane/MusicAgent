import json

import pytest

import feedback as fb
from conftest import event, make_session, write_json

C_MAJOR = fb.build_scale("C", "major")


# ─────────────────────────────────────────────
# Music theory helpers
# ─────────────────────────────────────────────

def test_build_scale():
    assert C_MAJOR == {0, 2, 4, 5, 7, 9, 11}
    assert 6 in fb.build_scale("G", "major")          # F#
    assert fb.build_scale("A", "minor") == C_MAJOR    # relative minor


def test_note_pc_strips_octave_and_sign():
    assert fb.note_pc("C#4") == 1
    assert fb.note_pc("B-1") == 11


# ─────────────────────────────────────────────
# Level 1 — notes
# ─────────────────────────────────────────────

def test_notes_in_and_out_of_scale():
    events = [event(0, [60]), event(1, [61]), event(2, [61]), event(3, [64])]
    res = fb.analyse_notes(events, C_MAJOR)
    assert res["total_notes"] == 4
    assert res["in_scale_count"] == 2
    assert res["score"] == 50
    assert res["frequent_errors"][0] == {"note": "C#", "count": 2}
    assert res["assessment"].startswith("Work again")
    assert res["avg_velocity"] == 80


def test_notes_legacy_confidence_is_read_as_velocity():
    ev = event(0, [60])
    del ev["notes"][0]["velocity"]
    ev["notes"][0]["confidence"] = 0.5
    assert fb.analyse_notes([ev], C_MAJOR)["avg_velocity"] == 63.5


def test_notes_no_data():
    assert fb.analyse_notes([], C_MAJOR)["status"] == "no_data"


# ─────────────────────────────────────────────
# Level 2 — rhythm
# ─────────────────────────────────────────────

def test_rhythm_steady(steady_session):
    res = fb.analyse_rhythm(steady_session["events"], 60)
    assert res["timing_score"] == 100
    assert res["timing_assessment"] == "Steady rhythm"
    assert res["hesitation_count"] == 0


def test_rhythm_irregular_scores_lower_than_steady(steady_session):
    times = [0, 0.85, 2.05, 2.9, 4.1, 4.95, 6.15, 7.0]
    irregular = [event(t, [60]) for t in times]
    steady = fb.analyse_rhythm(steady_session["events"], 60)["timing_score"]
    assert fb.analyse_rhythm(irregular, 60)["timing_score"] < steady


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation: deviation is measured against the nearest of 10 grid "
    "values, so it rarely exceeds 0.25 and very irregular playing is not flagged"))
def test_rhythm_very_irregular_is_flagged():
    times = [0, 1.25, 1.95, 3.2, 3.9, 5.15, 5.85, 7.1]
    res = fb.analyse_rhythm([event(t, [60]) for t in times], 60)
    assert res["timing_assessment"] == "rhythm too irregular"


def test_rhythm_skips_simultaneous_onsets(steady_session):
    events = steady_session["events"]
    doubled = sorted(events + [dict(e, time=e["time"] + 0.01) for e in events],
                     key=lambda e: e["time"])
    assert fb.analyse_rhythm(doubled, 60)["timing_score"] == 100


def test_rhythm_hesitation_detected_and_excluded_from_deviation():
    times = [0, 1, 2, 3, 8, 9, 10]          # 5 s gap at 60 BPM
    res = fb.analyse_rhythm([event(t, [60]) for t in times], 60)
    assert res["hesitation_count"] == 1
    assert res["hesitations"][0]["gap_beats"] == 5.0
    assert res["timing_score"] == 100


def test_rhythm_not_enough_intervals():
    res = fb.analyse_rhythm([event(0, [60]), event(1, [62])], 60)
    assert "timing_score" not in res
    assert "Not enough" in res["timing_assessment"]


def test_rhythm_no_data():
    assert fb.analyse_rhythm([], 60)["status"] == "no_data"


# ─────────────────────────────────────────────
# Level 3 — harmony
# ─────────────────────────────────────────────

def chords(*pairs):
    return [{"chord": c, "fit": f} for c, f in pairs]


def test_harmony_ignores_silence_segments():
    res = fb.analyse_harmony(
        chords(("N", "no_chord"), ("C", "in_scale"), ("F", "in_scale"),
               ("G", "in_scale"), ("F#:sus4", "out_of_scale"), ("N", "no_chord")),
        C_MAJOR, "C", "major")
    assert res["total_chord_segments"] == 4
    assert res["harmony_score"] == 75
    assert res["non_diatonic_chords"] == [{"chord": "F#:sus4", "count": 1}]


def test_harmony_only_silence_is_no_data():
    res = fb.analyse_harmony(chords(("N", "no_chord")), C_MAJOR, "C", "major")
    assert res["status"] == "no_data"


def test_detect_progressions():
    found = fb._detect_progressions(["C", "F", "G", "C", "G", "A:min", "F"], "C", "major")
    names = {f["progression"] for f in found}
    assert {"I-IV-V", "I-V-vi-IV"} <= names


# ─────────────────────────────────────────────
# Suggestions
# ─────────────────────────────────────────────

def test_suggestions_empty_session():
    sugg = fb.build_suggestions({"status": "no_data"}, {"status": "no_data"},
                                {"status": "no_data"}, 60)
    assert len(sugg) == 1 and "No notes" in sugg[0]["message"]


def test_suggestions_low_note_accuracy():
    notes = {"score": 40, "frequent_errors": [{"note": "C#", "count": 3}]}
    sugg = fb.build_suggestions(notes, {"timing_score": 100}, {}, 60)
    assert sugg[0]["priority"] == 1 and sugg[0]["area"] == "notes"
    assert "C#. Review" in sugg[0]["message"]


def test_suggestions_metronome_uses_session_bpm():
    sugg = fb.build_suggestions({"score": 95}, {"avg_ioi_deviation": 0.5,
                                                "timing_score": 50}, {}, 100)
    assert any("70 BPM" in s["message"] for s in sugg)


def test_suggestions_encouragement_needs_a_timing_score():
    no_timing = fb.build_suggestions({"score": 95}, {}, {}, 60)
    assert not any(s["area"] == "encouragement" for s in no_timing)
    good = fb.build_suggestions({"score": 95}, {"timing_score": 90}, {}, 60)
    assert any(s["area"] == "encouragement" for s in good)


# ─────────────────────────────────────────────
# End to end
# ─────────────────────────────────────────────

def test_main_writes_reports_and_auto_detects_chords(tmp_path, steady_session):
    session = write_json(tmp_path / "session_test.json", steady_session)
    write_json(tmp_path / "session_test_chords.json",
               {"chords": chords(("C", "in_scale"), ("G", "in_scale"))})

    fb.main([str(session)])

    out = json.loads((tmp_path / "feedback_session_test.json").read_text())
    assert out["note_accuracy"]["assessment"].startswith("Excellent")
    assert out["rhythm"]["timing_assessment"] == "Steady rhythm"
    assert out["harmony"]["harmony_score"] == 100
    txt = (tmp_path / "feedback_session_test.txt").read_text()
    assert "Assessment : Excellent" in txt
