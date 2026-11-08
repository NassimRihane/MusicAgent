import numpy as np
import pytest

import chord_evaluation as ce
import chord_rf
from conftest import SR, tones


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    return chord_rf.train(per_class=12, path=tmp_path_factory.mktemp("m") / "rf.joblib",
                          verbose=False)


def simple_render(piece, soundfont=None):
    """Stand-in for FluidSynth (no soundfont needed in tests): sine tones of the piece's notes."""
    end = max(e for _, _, e, _ in piece["notes"]) + 0.5
    audio = np.zeros(int(end * SR), dtype=np.float32)
    for pitch, start, stop, _ in piece["notes"]:
        y = tones([pitch], stop - start, amp=0.05)
        i = int(start * SR)
        audio[i:i + len(y)] += y[:len(audio) - i]
    return audio


@pytest.mark.parametrize("label,harte", [("C", "C:maj"), ("A:min", "A:min"), ("G:dom7", "G:7"),
                                         ("F:maj7", "F:maj7"), ("B:dim", "B:dim"), ("N", "N")])
def test_to_harte(label, harte):
    assert ce.to_harte(label) == harte


@pytest.mark.parametrize("texture", ce.TEXTURES)
def test_make_piece(texture):
    piece = ce.make_piece(np.random.default_rng(0), texture)
    assert len(piece["labels"]) == len(piece["intervals"]) and len(piece["labels"]) % 2 == 0
    assert np.all(np.diff(piece["intervals"][:, 0]) > 0)
    assert all(21 <= p <= 108 and e > s for p, s, e, _ in piece["notes"])
    assert all(":" in label for label in piece["labels"])


def test_progressions_only_use_the_model_vocabulary():
    qualities = {q for mode in ce.PROGRESSIONS.values() for prog in mode for _, q in prog}
    assert qualities <= set(chord_rf.QUALITIES) and qualities == set(chord_rf.QUALITIES)


def test_score_perfect_and_wrong_estimates():
    ref_i = np.array([[0.0, 2.0], [2.0, 4.0]])
    ref = ["C:maj", "G:7"]
    assert ce.score(ref_i, ref, ref_i, ref) == {"root": 1.0, "majmin": 1.0, "sevenths": 1.0}
    half = ce.score(ref_i, ref, ref_i, ["C:maj", "G:maj"])        # 7th missed
    assert half["root"] == 1.0 and half["majmin"] == 1.0 and half["sevenths"] == 0.5
    wrong = ce.score(ref_i, ref, ref_i, ["C:min", "D:maj"])
    assert wrong["root"] == 0.5 and wrong["majmin"] == 0.0


def test_evaluate_end_to_end(small_model, monkeypatch):
    import recognition as rc
    monkeypatch.setattr(ce, "render", simple_render)
    monkeypatch.setattr(rc, "transcribe", lambda path: [])   # music21 path then finds no chords
    result = ce.evaluate(pieces=3, seed=0, bundle=small_model, verbose=False)
    assert len(result["rows"]) == 6
    assert result["summary"]["rf"]["all"]["root"]["mean"] > 0.6   # sine-tone chords are easy
    assert result["summary"]["music21"]["all"]["root"]["mean"] == 0.0
    md = ce.markdown_report(result, heldout=0.9)
    assert "| Random Forest (audio chroma) | **all** |" in md and "90.0%" in md


def test_main_needs_the_soundfont(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "SOUNDFONT", tmp_path / "missing.sf2")
    with pytest.raises(SystemExit):
        ce.main([])


# ─────────────────────────────────────────────
# Real recordings
# ─────────────────────────────────────────────

def test_protocol_is_consistent():
    import recognition as rc
    from pathlib import Path
    protocol = ce.load_protocol()
    assert len(protocol) == 28 and len({c["label"] for c in protocol}) == 28
    for c in protocol:
        assert rc.chord_info(c["pitches"][1:])["label"] == c["label"], c["name"]
        assert c["pitches"][0] % 12 == c["pitches"][1] % 12       # left hand plays the root
    qualities = {c["label"].partition(":")[2] or "maj" for c in protocol}
    assert qualities == set(chord_rf.QUALITIES)
    sheet = (Path(ce.ROOT) / "protocols" / "chord_recording.md").read_text()
    for c in protocol:
        assert f"| {c['number']} | **{c['name']}** |" in sheet


def sine_take(protocol, hold=2.0, gap=1.0):
    notes = [(p, 0.5 + i * (hold + gap), 0.5 + i * (hold + gap) + hold, 80)
             for i, c in enumerate(protocol) for p in c["pitches"]]
    return simple_render({"notes": notes})


def test_segment_chords_finds_each_chord():
    protocol = ce.load_protocol()[:6]
    starts = ce.segment_chords(sine_take(protocol), len(protocol))
    assert np.allclose(starts, [0.5 + 3 * i for i in range(6)], atol=0.1)


def test_segment_chords_reports_missing_chords():
    with pytest.raises(ValueError, match="expected 28"):
        ce.segment_chords(sine_take(ce.load_protocol()[:3]), 28)


def test_segment_chords_without_silences_falls_back_to_onsets():
    protocol = ce.load_protocol()[:5]
    starts = ce.segment_chords(sine_take(protocol, hold=2.0, gap=0.0), len(protocol))
    assert len(starts) == 5


def test_score_per_chord_skips_excluded_chords():
    ref = ["C:maj", "C:aug", "G:7"]
    sc = ce.score_per_chord(ref, ["C:maj", "C:aug", "G:maj"])
    assert sc["root"] == 1.0 and sc["majmin"] == 1.0    # aug excluded from majmin
    assert sc["sevenths"] == 0.5                         # aug excluded, G7 → G wrong


def test_evaluate_recording_end_to_end(small_model, monkeypatch):
    import recognition as rc
    protocol = ce.load_protocol()[:8]
    audio = sine_take(protocol)
    played = [n for i, c in enumerate(protocol) for n in
              [{"start": 0.5 + 3 * i, "end": 2.5 + 3 * i, "midi": p, "velocity": 80}
               for p in c["pitches"]]]
    monkeypatch.setattr(rc, "transcribe", lambda path: played)   # a perfect transcription
    result = ce.evaluate_recording(audio, protocol, small_model)
    assert [c["number"] for c in result["chords"]] == list(range(1, 9))
    assert result["scores"]["music21"]["root"] == 1.0
    assert result["scores"]["rf"]["root"] >= 0.75
    md = ce.recording_report(result, "take.wav")
    assert "| 1 | C:maj |" in md
