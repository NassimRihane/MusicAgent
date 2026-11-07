# recognition.py
#
# Layer 1 — Standard music recognition
#
# Every analysis in this module is delegated to a widely used, published
# tool. Nothing here is specific to this project or trained by it:
#
#   note transcription : basic-pitch (Spotify; Bittner et al., ICASSP 2022)
#   key / scale        : music21 key analysis (Krumhansl–Schmuckler algorithm)
#   chord naming       : music21 chord symbols and Roman numeral analysis
#   beats / tempo      : librosa beat tracking (Ellis 2007) and tempo estimation
#   MIDI reading       : pretty_midi
#
# The only code of our own is glue: format conversion, grouping notes that
# start together, and collecting the notes that sound during each beat.
#
# A note is a dict: {"start": s, "end": s, "midi": int, "velocity": 0-127}



import numpy as np
import librosa



# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

SAMPLE_RATE  = 22050
CHORD_WINDOW = 0.05     # notes starting within 50 ms form one attack (chord)
ONSET_FPS    = 100      # frame rate of the onset envelope built from notes
TEMPO_EDGE_SEC = 4.0    # local tempo near the edges uses half-empty windows: ignored

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Rhythmic grid used to label note durations (in beats)
BEAT_GRID = {
    "whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "16th": 0.25,
    "dotted-half": 3.0, "dotted-quarter": 1.5, "dotted-eighth": 0.75,
    "quarter-triplet": 2.0 / 3.0, "eighth-triplet": 1.0 / 3.0,
}



# ─────────────────────────────────────────────
# 2. NOTE TRANSCRIPTION (basic-pitch, pretty_midi)
# ─────────────────────────────────────────────

_BP_MODEL = None

def get_basic_pitch_model():
    """Load the basic-pitch model once; reloading it on every call is slow."""
    global _BP_MODEL
    if _BP_MODEL is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


def transcribe(path, onset_thresh: float = 0.5, frame_thresh: float = 0.3,
               min_note_dur: float = 0.05) -> list:
    """Transcribe an audio file with basic-pitch. Returns notes sorted by onset."""
    import contextlib, io
    from basic_pitch.inference import predict
    with contextlib.redirect_stdout(io.StringIO()):   # basic-pitch prints debug lines
        _, midi_data, _ = predict(str(path), get_basic_pitch_model(),
                                  onset_threshold=onset_thresh,
                                  frame_threshold=frame_thresh,
                                  minimum_note_length=min_note_dur * 1000)
    return midi_to_notes(midi_data)


def midi_to_notes(pm, sustain: bool = False) -> list:
    """
    Notes from a pretty_midi object. With sustain=True, notes are held while
    the sustain pedal (CC 64 >= 64) is down, as they sound in a recording.
    """
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        pedal = _pedal_intervals(inst.control_changes) if sustain else []
        for n in inst.notes:
            end = n.end
            for down, up in pedal:
                if down <= end < up:
                    end = up
                    break
            notes.append({"start": float(n.start), "end": float(end),
                          "midi": int(n.pitch), "velocity": int(n.velocity)})
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    if sustain:
        _cut_repeated_notes(notes)
    return notes


def load_midi_notes(path, sustain: bool = True) -> list:
    import pretty_midi
    return midi_to_notes(pretty_midi.PrettyMIDI(str(path)), sustain=sustain)


def _pedal_intervals(control_changes) -> list:
    """(down, up) times of the sustain pedal."""
    intervals, down = [], None
    for cc in sorted((c for c in control_changes if c.number == 64), key=lambda c: c.time):
        if cc.value >= 64 and down is None:
            down = cc.time
        elif cc.value < 64 and down is not None:
            intervals.append((down, cc.time))
            down = None
    if down is not None:
        intervals.append((down, float("inf")))
    return intervals


def _cut_repeated_notes(notes: list):
    """A re-struck pitch ends the previous sounding note of the same pitch."""
    last = {}
    for n in notes:
        prev = last.get(n["midi"])
        if prev is not None and prev["end"] > n["start"]:
            prev["end"] = n["start"]
        last[n["midi"]] = n
    for n in notes:
        if n["end"] == float("inf"):
            n["end"] = n["start"] + 1.0



# ─────────────────────────────────────────────
# 3. NOTE HELPERS
# ─────────────────────────────────────────────

def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def onset_groups(notes: list, window: float = CHORD_WINDOW) -> list:
    """Group notes whose onsets are within `window` seconds into single attacks."""
    groups = []
    for n in sorted(notes, key=lambda n: (n["start"], n["midi"])):
        if groups and n["start"] - groups[-1][0]["start"] <= window:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def quantize_duration(duration_sec: float, bpm: float) -> str:
    """Nearest note value in BEAT_GRID for a duration in seconds."""
    beats = duration_sec / (60.0 / bpm)
    return min(BEAT_GRID, key=lambda name: abs(beats - BEAT_GRID[name]))


def notes_to_events(notes: list, scale_pc: set = None, bpm: float = 60.0) -> list:
    """
    Session events: one event per attack (single note or chord), with its
    onset time, mean duration, rhythm label and in-scale flags.
    A pitch reported twice in one attack is kept once (the loudest).
    """
    events = []
    for group in onset_groups(notes):
        onset = min(n["start"] for n in group)
        by_pitch = {}
        for n in group:
            if n["midi"] not in by_pitch or n["velocity"] > by_pitch[n["midi"]]["velocity"]:
                by_pitch[n["midi"]] = n
        group = sorted(by_pitch.values(), key=lambda n: n["midi"])

        note_dicts = []
        for n in group:
            pc = n["midi"] % 12
            note_dicts.append({
                "midi":        n["midi"],
                "name":        note_name(n["midi"]),
                "pitch_class": pc,
                "in_scale":    pc in scale_pc if scale_pc is not None else None,
                "velocity":    int(n["velocity"]),
            })
        duration = float(np.mean([n["end"] - n["start"] for n in group]))
        events.append({
            "type":     "chord" if len(group) > 1 else "note",
            "notes":    note_dicts,
            "duration": round(duration, 3),
            "rhythm":   quantize_duration(duration, bpm),
            "in_scale": all(n["in_scale"] for n in note_dicts) if scale_pc is not None else None,
            "time":     round(onset, 3),
        })
    return events


def session_notes(session: dict) -> list:
    """Notes of a session JSON: the "notes" list, or rebuilt from events (older sessions)."""
    if session.get("notes"):
        return session["notes"]
    notes = []
    for e in session.get("events", []):
        start = float(e.get("time", 0.0))
        end = start + float(e.get("duration") or 0.1)
        for n in e.get("notes", []):
            vel = n.get("velocity")
            if vel is None:   # sessions before 2026-10-10 stored velocity/127 as "confidence"
                vel = n.get("confidence", 0.63) * 127
            notes.append({"start": start, "end": end, "midi": int(n["midi"]), "velocity": float(vel)})
    return notes



# ─────────────────────────────────────────────
# 4. KEY AND SCALE (music21)
# ─────────────────────────────────────────────

def _m21_tonic(name: str) -> str:
    """'Bb' → 'B-' (music21 spelling of flats)."""
    return name[0] + name[1:].replace("b", "-")


def _display_tonic(name: str) -> str:
    """'B-' → 'Bb'."""
    return name.replace("-", "b")


def parse_key(key_name: str):
    """'C major', 'A minor', 'F# major', 'Bb minor' → music21 Key."""
    from music21 import key
    parts = key_name.split()
    tonic = parts[0] if parts else "C"
    mode = parts[1] if len(parts) > 1 else "major"
    return key.Key(_m21_tonic(tonic), mode)


def scale_pitch_classes(key_name: str) -> set:
    """
    Pitch classes that belong to the key, via music21. For minor keys this is
    the union of the natural, harmonic and melodic minor scales, since the
    raised 6th and 7th degrees are normal in minor.
    """
    from music21 import scale
    k = parse_key(key_name)
    scales = [k.getScale()]
    if k.mode == "minor":
        scales += [scale.HarmonicMinorScale(k.tonic), scale.MelodicMinorScale(k.tonic)]
    return {p.pitchClass for sc in scales for p in sc.getPitches()}


def detect_key(notes: list) -> dict:
    """
    Key of a performance with music21's Krumhansl–Schmuckler analysis
    (pitch classes weighted by duration).
    major_minor = best major correlation − best minor correlation
    (> 0 leans major, < 0 leans minor).
    """
    from music21 import stream, note
    if not notes:
        return {"key": None, "mode": None, "correlation": None, "major_minor": 0.0}
    s = stream.Stream()
    for n in notes:
        m = note.Note(int(n["midi"]))
        m.quarterLength = max(n["end"] - n["start"], 0.01)
        s.append(m)
    k = s.analyze("key")
    interpretations = [k] + list(k.alternateInterpretations)
    best = {mode: max(i.correlationCoefficient for i in interpretations if i.mode == mode)
            for mode in ("major", "minor")}
    return {"key": f"{_display_tonic(k.tonic.name)} {k.mode}", "mode": k.mode,
            "correlation": round(float(k.correlationCoefficient), 3),
            "major_minor": round(float(best["major"] - best["minor"]), 3)}



# ─────────────────────────────────────────────
# 5. BEATS AND TEMPO (librosa)
# ─────────────────────────────────────────────

def onset_envelope_from_notes(notes: list, fps: int = ONSET_FPS) -> np.ndarray:
    """Onset-strength curve built from note onsets (one impulse per attack)."""
    if not notes:
        return np.zeros(fps)
    env = np.zeros(int(max(n["end"] for n in notes) * fps) + fps)
    for g in onset_groups(notes):
        env[int(g[0]["start"] * fps)] += sum(n["velocity"] for n in g) / 127.0
    return np.convolve(env, np.hanning(5), mode="same")


def track_beats(notes: list = None, audio: np.ndarray = None, sr: int = SAMPLE_RATE) -> dict:
    """
    Beat times and tempo with librosa, from audio if given, else from notes.

    tempo_cv: tempo stability, the coefficient of variation of librosa's local
    tempo estimate (0 = perfectly steady). Frames within TEMPO_EDGE_SEC of the
    edges are ignored (incomplete analysis windows), and each local estimate is
    folded to the octave of the median, as in standard tempo evaluation
    (MIREX), so that 100 vs 200 BPM ambiguities don't count as instability.
    """
    if audio is not None:
        env, rate, hop = librosa.onset.onset_strength(y=audio, sr=sr), sr, 512
    else:
        env, rate, hop = onset_envelope_from_notes(notes or []), ONSET_FPS, 1
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=rate, hop_length=hop, units="time")
    local = librosa.feature.tempo(onset_envelope=env, sr=rate, hop_length=hop, aggregate=None)
    edge = int(TEMPO_EDGE_SEC * rate / hop)
    if len(local) > 3 * edge:
        local = local[edge:-edge]
    local = local[local > 0]
    cv = 0.0
    if len(local):
        local = local * 2.0 ** np.round(np.log2(np.median(local) / local))
        cv = float(np.std(local) / np.mean(local))
    return {"tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1),
            "beat_times": [round(float(b), 3) for b in beats],
            "tempo_cv": round(cv, 3)}



# ─────────────────────────────────────────────
# 6. CHORDS (music21)
# ─────────────────────────────────────────────

CHORD_MIN_WEIGHT = 0.2   # a pitch class counts if it has ≥ 20 % of the strongest one's weight
CHORD_MAX_NOTES  = 4
CHORD_MAX_BEATS  = 3     # a window grows up to 3 beats until it holds 3 pitch classes

def chord_info(midis: list, key_name: str = None) -> dict:
    """
    Name a set of pitches with music21: chord symbol (e.g. 'Am7', 'C/E'),
    root, quality, canonical label ('A:min7') and Roman numeral in the key.
    """
    from music21 import chord, harmony, roman
    c = chord.Chord(sorted(set(midis)))
    c.simplifyEnharmonics(inPlace=True)
    try:
        symbol = harmony.chordSymbolFigureFromChord(c)
    except Exception:
        symbol = "Chord Symbol Cannot Be Identified"
    if "Cannot" in symbol:
        symbol = c.pitchedCommonName
    else:   # lead-sheet spelling: 'B-m' → 'Bbm', 'Cpower' → 'C5'
        symbol = symbol.replace("-", "b").replace("power", "5")
    info = {"chord": symbol,
            "root": _display_tonic(c.root().name), "quality": c.quality,
            "label": canonical_label(c)}
    if key_name:
        rn = roman.romanNumeralFromChord(c, parse_key(key_name))
        scale = scale_pitch_classes(key_name)
        pcs = {p.pitchClass for p in c.pitches}
        inside = len(pcs & scale) / len(pcs)
        info.update({"roman": rn.figure, "numeral": rn.romanNumeralAlone,
                     "fit": "in_scale" if inside == 1 else "partial" if inside >= 0.5
                            else "out_of_scale"})
    return info


SEVENTH_NAMES = {"dominant seventh chord": "dom7", "major seventh chord": "maj7",
                 "minor seventh chord": "min7"}
QUALITY_NAMES = {"major": "maj", "minor": "min", "diminished": "dim", "augmented": "aug"}

def canonical_label(c) -> str:
    """Compact label shared with the Random Forest chord model: 'C', 'A:min', 'G:dom7'."""
    root = NOTE_NAMES[c.root().pitchClass]
    quality = SEVENTH_NAMES.get(c.commonName) or QUALITY_NAMES.get(c.quality)
    if quality is None:
        return "X"   # not in the triad/seventh vocabulary
    return root if quality == "maj" else f"{root}:{quality}"


def _window_pitches(notes: list, start: float, end: float):
    """Pitch-class weights (overlap × velocity) and lowest pitch per class in [start, end)."""
    weight, lowest = np.zeros(12), {}
    for n in notes:
        overlap = min(n["end"], end) - max(n["start"], start)
        if overlap > 0:
            pc = n["midi"] % 12
            weight[pc] += overlap * n["velocity"]
            lowest[pc] = min(lowest.get(pc, 128), n["midi"])
    keep = [int(pc) for pc in np.argsort(weight)[::-1][:CHORD_MAX_NOTES]
            if weight[pc] > 0 and weight[pc] >= CHORD_MIN_WEIGHT * weight.max()]
    return keep, lowest


def chords_from_notes(notes: list, beat_times: list, key_name: str = None) -> list:
    """
    Chords named by music21 from the notes sounding in each beat (weighted by
    how long and how loud they sound). A beat with fewer than three pitch
    classes is extended over the next beats (up to CHORD_MAX_BEATS) so that a
    chord, not an interval, is named. Identical consecutive chords are merged.
    """
    if not notes or len(beat_times) < 2:
        return []
    bounds = list(beat_times) + [beat_times[-1] + (beat_times[-1] - beat_times[-2])]
    beats, i = [], 0
    while i < len(bounds) - 1:
        j = i + 1
        keep, lowest = _window_pitches(notes, bounds[i], bounds[j])
        while keep and len(keep) < 3 and j - i < CHORD_MAX_BEATS and j < len(bounds) - 1:
            j += 1
            keep, lowest = _window_pitches(notes, bounds[i], bounds[j])
        if not keep:
            beats.append({"start": bounds[i], "end": bounds[j], "chord": "N", "label": "N",
                          "fit": "no_chord"})
        else:
            info = chord_info([lowest[pc] for pc in keep], key_name)
            beats.append({"start": bounds[i], "end": bounds[j], **info})
        i = j

    segments = []
    for b in beats:
        if segments and segments[-1]["chord"] == b["chord"]:
            segments[-1]["end"] = b["end"]
        else:
            segments.append(dict(b))
    for s in segments:
        s["start"], s["end"] = round(s["start"], 3), round(s["end"], 3)
        s["method"] = "music21"
    return segments
