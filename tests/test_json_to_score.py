import pytest

import json_to_score as js
from conftest import event, make_session, write_json


def score_notes(session):
    return list(js.build_score(session, "Test").parts[0].flatten().notes)


@pytest.mark.parametrize("rhythm,quarter_length", [
    ("quarter", 1.0), ("half", 2.0), ("dotted-quarter", 1.5),
    ("eighth", 0.5), ("eighth-triplet", 1 / 3),
])
def test_rhythm_labels_become_durations(rhythm, quarter_length):
    notes = score_notes(make_session([event(0, [60], rhythm=rhythm)]))
    assert float(notes[0].duration.quarterLength) == pytest.approx(quarter_length)


def test_notes_and_chords():
    notes = score_notes(make_session([event(0, [60]), event(1, [60, 64, 67])]))
    assert notes[0].nameWithOctave == "C4"
    assert [p.nameWithOctave for p in notes[1].pitches] == ["C4", "E4", "G4"]


def test_missing_rhythm_falls_back_to_raw_duration():
    ev = event(0, [60], duration=2.0)
    del ev["rhythm"]
    assert float(score_notes(make_session([ev], bpm=60))[0].duration.quarterLength) == 2.0


def test_legacy_value_format():
    session = make_session([{"type": "note", "value": "D4", "rhythm": "noire"},
                            {"type": "chord", "value": "Amin", "rhythm": "blanche"}])
    notes = score_notes(session)
    assert notes[0].nameWithOctave == "D4"
    assert [p.name for p in notes[1].pitches] == ["A", "C", "E"]


def test_chord_label_to_pitches():
    assert js._chord_label_to_pitches("G") == ["G4", "B4", "D5"]
    assert js._chord_label_to_pitches("Bdim") == ["B4", "D5", "F5"]
    assert js._chord_label_to_pitches("?") == []


def test_main_exports_musicxml(tmp_path):
    session = write_json(tmp_path / "session_test.json",
                         make_session([event(0, [60]), event(1, [62])]))
    js.main([str(session)])
    assert (tmp_path / "session_test.musicxml").stat().st_size > 0
