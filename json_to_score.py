# json_to_score.py
#
# Step 1b — Session JSON → Sheet music
# Converts a transcription session file into a readable score.
#
# Dependencies: music21
#   pip install music21
#
# For PDF output, LilyPond must also be installed (free):
#   https://lilypond.org/download.html
#   music21 will find it automatically if it is on your PATH.
#
# Usage:
#   python json_to_score.py sessions/session_20240101_120000.json
#   python json_to_score.py sessions/session_20240101_120000.json --output pdf
#   python json_to_score.py sessions/session_20240101_120000.json --output midi

import argparse
import json
import sys
from pathlib import Path

from music21 import stream, note, chord, meter, tempo, key, metadata, environment



# Arguments


parser = argparse.ArgumentParser(description="Convert a session JSON file to sheet music")
parser.add_argument("session",  help="Path to the session JSON file")
parser.add_argument("--output", default="musicxml",
                    choices=["musicxml", "pdf", "midi", "all"],
                    help="Output format (default: musicxml)")
parser.add_argument("--title",  default=None,
                    help="Score title (defaults to the session filename)")
args = parser.parse_args()



# Rhythmic value
# Maps the quantised rhythm strings from transcription.py to music21 duration type strings.


# Dotted values and triplets require special handling in music21 (see build_score).
RHYTHM_TO_DURATION = {
    # Standard values 
    "whole":            ("whole",   False, False),   # (type, dotted, triplet)
    "half":             ("half",    False, False),
    "quarter":          ("quarter", False, False),
    "eighth":           ("eighth",  False, False),
    "16th":             ("16th",    False, False),
    # Dotted values
    "dotted-half":      ("half",    True,  False),
    "dotted-quarter":   ("quarter", True,  False),
    "dotted-eighth":    ("eighth",  True,  False),
    # Triplets
    "quarter-triplet":  ("quarter", False, True),
    "eighth-triplet":   ("eighth",  False, True),
    #  Legacy French labels (older session files) 
    "ronde":            ("whole",   False, False),
    "blanche":          ("half",    False, False),
    "noire":            ("quarter", False, False),
}

DEFAULT_DURATION = ("quarter", False, False)   # fallback when label is missing



# Load Session


def load_session(path: str) -> dict:
    """Load and validate a session JSON file."""
    session_path = Path(path)
    if not session_path.exists():
        print(f"Error: file not found — {path}")
        sys.exit(1)

    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "events" not in data:
        print("Error: the JSON file does not contain an 'events' key.")
        sys.exit(1)

    return data



# Build Music21 score


def build_score(session: dict, title: str) -> stream.Score:
    """
    Convert a session dict into a music21 Score object.

    Notes and chords are placed sequentially in a single Part.
    Duration is taken from the quantised 'rhythm' field; if absent,
    the raw 'duration' (seconds) is converted using the session BPM.
    """
    meta = session.get("meta", {})
    bpm  = float(meta.get("bpm", 60))
    scale_str = meta.get("scale", "C major")

    #  Score metadata 
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title     = title
    score.metadata.composer  = "Transcribed by transcription.py"

    #  Single piano part 
    part = stream.Part()
    part.partName = "Piano"

    #  Time signature & tempo 
    part.append(meter.TimeSignature("4/4"))
    mm = tempo.MetronomeMark(number=bpm)
    part.append(mm)

    #  Key signature (best-effort parse from "C major" / "G minor") 
    try:
        root, mode = scale_str.split()
        ks = key.Key(root, mode)
        part.append(ks)
    except Exception:
        part.append(key.Key("C"))   # fallback: C major

    #  Convert events 
    beat_duration_sec = 60.0 / bpm   # seconds per quarter note

    skipped = 0
    for event in session["events"]:
        event_type = event.get("type")

        # Determine duration — returns (type_str, dotted, triplet)
        rhythm_label = event.get("rhythm")
        if rhythm_label and rhythm_label in RHYTHM_TO_DURATION:
            dur_tuple = RHYTHM_TO_DURATION[rhythm_label]
        elif event.get("duration") is not None:
            raw_beats = event["duration"] / beat_duration_sec
            dur_tuple = _beats_to_duration_type(raw_beats)
        else:
            dur_tuple = DEFAULT_DURATION

        dur_type, dur_dotted, dur_triplet = dur_tuple

        # Build the music21 element
        # New format (basic-pitch): events have a 'notes' list of dicts with 'name' key.
        # Legacy format (aubio): note events have a 'value' string, chord events have 'value'.
        note_list = event.get("notes")   # new format

        if note_list:
            #  New format (basic-pitch) 
            pitch_strs = [n["name"] for n in note_list if "name" in n]
            if not pitch_strs:
                skipped += 1
                continue
            try:
                if len(pitch_strs) == 1:
                    n = note.Note(pitch_strs[0])
                    _apply_duration(n, dur_type, dur_dotted, dur_triplet)
                    part.append(n)
                else:
                    c = chord.Chord(pitch_strs)
                    _apply_duration(c, dur_type, dur_dotted, dur_triplet)
                    part.append(c)
            except Exception as exc:
                print(f"  Warning: could not create element {pitch_strs} — {exc}")
                skipped += 1

        elif event_type == "note":
            #  Legacy format (aubio/old sessions) 
            pitch_str = event.get("value", "")
            if not pitch_str:
                skipped += 1
                continue
            try:
                n = note.Note(pitch_str)
                _apply_duration(n, dur_type, dur_dotted, dur_triplet)
                cents_off = event.get("cents_off")
                if cents_off is not None:
                    n.editorial.cents_deviation = cents_off
                part.append(n)
            except Exception as exc:
                print(f"  Warning: could not create note '{pitch_str}' — {exc}")
                skipped += 1

        elif event_type == "chord":
            #  Legacy format 
            chord_label = event.get("value", "")
            if not chord_label:
                skipped += 1
                continue
            try:
                pitches = _chord_label_to_pitches(chord_label)
                if pitches:
                    c = chord.Chord(pitches)
                    _apply_duration(c, dur_type, dur_dotted, dur_triplet)
                    part.append(c)
                else:
                    skipped += 1
            except Exception as exc:
                print(f"  Warning: could not create chord '{chord_label}' — {exc}")
                skipped += 1

        else:
            skipped += 1   # unknown event type

    score.append(part)

    if skipped:
        print(f"  ({skipped} event(s) skipped due to missing or invalid data)")

    return score



# 4. HELPERS


def _apply_duration(element, dur_type: str, dotted: bool, triplet: bool):
    """
    Apply a duration to a music21 note or chord, handling dotted values
    and triplets correctly via music21's Duration API.
    """
    from music21 import duration as m21dur

    if triplet:
        # Triplet: three notes fit in the space of two of the same type.
        # music21 represents this with a Tuplet object.
        d = m21dur.Duration(type=dur_type)
        t = m21dur.Tuplet(3, 2)          # 3 in the time of 2
        t.bracket = True
        d.appendTuplet(t)
        element.duration = d
    elif dotted:
        element.duration = m21dur.Duration(type=dur_type)
        element.duration.dots = 1
    else:
        element.duration.type = dur_type


def _beats_to_duration_type(beats: float) -> tuple:
    """
    Snap a beat count to the nearest entry in the full rhythm grid.
    Returns a (type_str, dotted, triplet) tuple matching RHYTHM_TO_DURATION.
    """
    # Mirror of BEAT_GRID from transcription.py
    grid = {
        "whole":           (4.0,         "whole",   False, False),
        "half":            (2.0,         "half",    False, False),
        "quarter":         (1.0,         "quarter", False, False),
        "eighth":          (0.5,         "eighth",  False, False),
        "16th":            (0.25,        "16th",    False, False),
        "dotted-half":     (3.0,         "half",    True,  False),
        "dotted-quarter":  (1.5,         "quarter", True,  False),
        "dotted-eighth":   (0.75,        "eighth",  True,  False),
        "quarter-triplet": (2.0 / 3.0,  "quarter", False, True),
        "eighth-triplet":  (1.0 / 3.0,  "eighth",  False, True),
    }
    best_tuple, best_diff = ("quarter", False, False), float("inf")
    for _, (value, dtype, dotted, triplet) in grid.items():
        diff = abs(beats - value)
        if diff < best_diff:
            best_diff = diff
            best_tuple = (dtype, dotted, triplet)
    return best_tuple


# Chord intervals in semitones from the root (mirrors transcription.py)
_CHORD_INTERVALS = {
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    # "maj" is the fallback (no suffix)
}

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _chord_label_to_pitches(label: str) -> list:
    """
    Parse a chord label such as "C", "Amin", "Bdim", "Gmaj7"
    and return a list of pitch strings in the 4th octave, e.g. ["C4", "E4", "G4"].
    Returns an empty list if the label cannot be parsed.
    """
    # Identify the root (one or two characters: e.g. "C", "C#", "Bb")
    root_name = None
    suffix    = ""
    for candidate in sorted(_NOTE_NAMES, key=len, reverse=True):   # longest first
        if label.startswith(candidate):
            root_name = candidate
            suffix    = label[len(candidate):]
            break

    if root_name is None:
        return []

    root_idx = _NOTE_NAMES.index(root_name)

    # Match suffix to chord type
    intervals = None
    for chord_type, ivs in _CHORD_INTERVALS.items():
        if suffix == chord_type:
            intervals = ivs
            break
    if intervals is None:
        intervals = [0, 4, 7]   # default: major triad

    # Build pitch strings (anchor to octave 4)
    pitches = []
    for semitones in intervals:
        pc     = (root_idx + semitones) % 12
        octave = 4 + (root_idx + semitones) // 12
        pitches.append(f"{_NOTE_NAMES[pc]}{octave}")

    return pitches



# Export


def export_score(score: stream.Score, base_path: Path, output_format: str):
    """Write the score to disk in the requested format(s)."""
    formats = ["musicxml", "pdf", "midi"] if output_format == "all" else [output_format]

    for fmt in formats:
        if fmt == "musicxml":
            out_path = base_path.with_suffix(".musicxml")
            score.write("musicxml", fp=str(out_path))
            print(f"  MusicXML saved : {out_path}")
            print(f"  → Open with MuseScore, Sibelius, Dorico, or any notation software.")

        elif fmt == "pdf":
            out_path = base_path.with_suffix(".pdf")
            lily_fp = str(base_path)
            try:
                score.write("lily.pdf", fp=lily_fp)
                print(f"  PDF saved      : {out_path}")
            except Exception as exc:
                print(f"  PDF export failed: {exc}")
                print("  Make sure LilyPond is installed: https://lilypond.org/download.html")

        elif fmt == "midi":
            out_path = base_path.with_suffix(".mid")
            score.write("midi", fp=str(out_path))
            print(f"  MIDI saved     : {out_path}")



# Main


def main():
    session_path = Path(args.session)
    session      = load_session(str(session_path))
    meta         = session.get("meta", {})

    # Score title
    title = args.title or session_path.stem.replace("_", " ").title()

    print("=" * 60)
    print(f"  JSON → Score converter")
    print(f"  File  : {session_path.name}")
    print(f"  Scale : {meta.get('scale', '?')}  |  BPM: {meta.get('bpm', '?')}")
    print(f"  Events: {len(session['events'])}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    # Build the score
    score = build_score(session, title)

    # Output path: same folder as the JSON, same stem
    base_output_path = session_path.with_suffix("")
    export_score(score, base_output_path, args.output)

    print("Done.")


if __name__ == "__main__":
    main()