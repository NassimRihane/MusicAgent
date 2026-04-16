# transcription.py
#
# Step 1 — Real-time transcription for piano
# Pipeline: audio capture → onset/silence detection → basic-pitch → buffer → display
#
# Dependencies: sounddevice, numpy, librosa, basic-pitch, music21
# Install:  pip install sounddevice numpy librosa basic-pitch music21
#
# Usage:
#   python transcription.py --scale C --type major --bpm 60
#   python transcription.py --list-devices
#   python transcription.py --device 2 --scale G --type minor --bpm 80

import argparse
import queue
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa


# ─────────────────────────────────────────────
# 0. COMMAND-LINE ARGUMENTS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Real-time piano transcription")
parser.add_argument("--scale",        default="C",
                    help="Root note of the scale (e.g. C, G, F#)")
parser.add_argument("--type",         default="major",
                    choices=["major", "minor"], help="Scale type")
parser.add_argument("--bpm",          type=float, default=60.0,
                    help="Tempo in BPM")
parser.add_argument("--device",       type=int, default=None,
                    help="Index of the audio input device (see --list-devices)")
parser.add_argument("--list-devices", action="store_true",
                    help="List all available audio input devices and exit")
parser.add_argument("--min-note-dur", type=float, default=0.05,
                    help="Minimum note duration accepted from basic-pitch (seconds, default 0.05)")
parser.add_argument("--onset-thresh", type=float, default=0.5,
                    help="basic-pitch onset sensitivity 0–1 (default 0.5, lower = more notes)")
parser.add_argument("--frame-thresh", type=float, default=0.3,
                    help="basic-pitch frame sensitivity 0–1 (default 0.3)")
args = parser.parse_args()

if args.list_devices:
    print("\nAvailable audio input devices:\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marker = " ◄ (default)" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}]  {d['name']}{marker}")
    print("\nUse --device <index> to select a microphone.")
    exit(0)


# ─────────────────────────────────────────────
# 1. CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────

SAMPLE_RATE = 22050   # basic-pitch works natively at 22050 Hz
HOP_SIZE    = 512     # audio callback block size (~23 ms per hop)

# Onset / silence detection (RMS-based, calibrated at startup)
RMS_ONSET_THRESHOLD   = 0.02   # overwritten after calibration
RMS_SILENCE_THRESHOLD = 0.005  # overwritten after calibration
ONSET_COOLDOWN_SEC    = 0.15   # minimum gap between two onsets

# Segment collection
# We collect audio until silence, then send the whole segment to basic-pitch.
# MIN_SEGMENT_SEC: segment must be at least this long before analysis.
# MAX_SEGMENT_SEC: hard cap — send for analysis even if no silence detected.
# SILENCE_HOLD_SEC: silence must persist for this long before we declare end-of-note.
#   This avoids triggering on brief gaps within a legato phrase.
MIN_SEGMENT_SEC  = 0.4    # raised from 0.2 — filters very short noise bursts
MAX_SEGMENT_SEC  = 4.0
SILENCE_HOLD_SEC = 0.15   # silence must last 150ms before triggering analysis

# Rhythmic quantisation grid (in quarter-note beat units)
# Dotted values = 1.5× base; triplets = 2/3 of next longer value.
BEAT_GRID = {
    "whole":           4.0,
    "half":            2.0,
    "quarter":         1.0,
    "eighth":          0.5,
    "16th":            0.25,
    "dotted-half":     3.0,
    "dotted-quarter":  1.5,
    "dotted-eighth":   0.75,
    "quarter-triplet": 2.0 / 3.0,
    "eighth-triplet":  1.0 / 3.0,
}


# ─────────────────────────────────────────────
# 2. SCALES
# ─────────────────────────────────────────────

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F',
                    'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_NAME_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
    'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],   # natural minor
}

def build_scale(root_name: str, scale_type: str) -> set:
    root_pc   = NOTE_NAME_TO_PC[root_name]
    intervals = SCALE_INTERVALS[scale_type]
    return set((root_pc + i) % 12 for i in intervals)

scale_pc = build_scale(args.scale, args.type)


# ─────────────────────────────────────────────
# 3. NOTE / FREQUENCY CONVERTERS
# ─────────────────────────────────────────────

A4_FREQ = 440.0

def freq_to_midi(f: float) -> float:
    return 69 + 12 * np.log2(f / A4_FREQ)

def midi_to_note_name(midi_int: int) -> str:
    pc     = midi_int % 12
    octave = midi_int // 12 - 1
    return f"{NOTE_NAMES_SHARP[pc]}{octave}"

def note_name_to_pc(name: str) -> int:
    """Extract pitch class from a note name like 'C#4' or 'Bb3'."""
    base = ''.join(c for c in name if not c.isdigit() and c != '-')
    return NOTE_NAME_TO_PC.get(base, 0)


# ─────────────────────────────────────────────
# 4. RHYTHMIC QUANTISATION
# ─────────────────────────────────────────────

def quantize_duration(duration_sec: float, bpm: float) -> str:
    """
    Convert a raw duration (seconds) to the nearest note value in BEAT_GRID.
    Returns a key from BEAT_GRID, e.g. 'eighth', 'dotted-quarter', etc.
    """
    beat_duration  = 60.0 / bpm
    duration_beats = duration_sec / beat_duration

    best_name = "quarter"
    best_diff = float("inf")
    for name, beats in BEAT_GRID.items():
        diff = abs(duration_beats - beats)
        if diff < best_diff:
            best_diff = diff
            best_name = name
    return best_name


# ─────────────────────────────────────────────
# 5. TRANSCRIPTION BUFFER
# ─────────────────────────────────────────────

class TranscriptionBuffer:
    """
    Stores musical events detected in real time.
    Each event is a dict:
        time        float   timestamp from session start (seconds)
        type        str     "note" or "chord"
        notes       list    list of note dicts (see below) — chords have >1
        duration    float   raw duration in seconds
        rhythm      str     quantised note value
        in_scale    bool    True if ALL notes are in the current scale

    Each note dict inside 'notes':
        midi        int     MIDI number
        name        str     e.g. "C4"
        pitch_class int     0–11
        in_scale    bool
        confidence  float   basic-pitch confidence
    """

    def __init__(self):
        self.events     = []
        self.start_time = time.time()

    def add(self, event: dict):
        event["time"] = round(event["time"] - self.start_time, 3)
        self.events.append(event)

    def to_json(self, path=None) -> dict:
        data = {
            "meta": {
                "scale": f"{args.scale} {args.type}",
                "bpm":   args.bpm,
                "date":  datetime.now().isoformat(),
            },
            "events": self.events,
        }
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return data

    def display_last(self):
        """Print the most recent event to the console."""
        if not self.events:
            return
        e         = self.events[-1]
        timestamp = f"[{e['time']:6.2f}s]"
        rhythm    = e.get("rhythm", "?")
        in_scale  = e.get("in_scale", False)
        marker    = "✓" if in_scale else "✗"

        note_list = e.get("notes", [])
        names     = " + ".join(n["name"] for n in note_list)
        ev_type   = "CHORD" if len(note_list) > 1 else "NOTE "

        print(
            f"{timestamp}  {ev_type}  {names:20s} | "
            f"{rhythm:16s} | "
            f"{'in scale' if in_scale else 'out of scale':13s} {marker}"
        )


# ─────────────────────────────────────────────
# 6. NOISE FLOOR CALIBRATION
# ─────────────────────────────────────────────

def calibrate_noise_floor(device, duration_sec: float = 2.0):
    """
    Record ambient noise and derive dynamic RMS onset/silence thresholds.
    Returns (onset_threshold, silence_threshold).
    """
    print(f"  Calibrating noise floor ({duration_sec:.0f}s) — please do not play...")
    n_samples = int(SAMPLE_RATE * duration_sec)
    recording = sd.rec(n_samples, samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=device)
    sd.wait()
    audio = recording.flatten()

    rms_values = []
    for i in range(0, len(audio) - HOP_SIZE, HOP_SIZE):
        chunk = audio[i : i + HOP_SIZE]
        rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))

    noise_rms         = float(np.percentile(rms_values, 95))

    # Multipliers explanation:
    #   ×10 for onset   — requires the signal to be 10× louder than background noise.
    #                     This prevents the mic's self-noise from triggering analysis.
    #   ×3  for silence — once a note is playing, RMS must drop to 3× noise floor
    #                     before we consider it finished.
    # If the room is very quiet (noise_rms < 0.001), we apply a hard floor
    # so the thresholds remain meaningful.
    noise_rms_floored = max(noise_rms, 0.001)
    onset_threshold   = noise_rms_floored * 10.0
    silence_threshold = noise_rms_floored * 3.0

    print(f"  Noise floor: RMS={noise_rms:.5f} (floored={noise_rms_floored:.5f}) → "
          f"onset={onset_threshold:.5f} / silence={silence_threshold:.5f}")
    return onset_threshold, silence_threshold


# ─────────────────────────────────────────────
# 7. BASIC-PITCH ANALYSIS
# ─────────────────────────────────────────────

def analyse_segment(audio_segment: np.ndarray, segment_duration: float) -> list:
    """
    Run basic-pitch on an audio segment and return a list of events.

    basic-pitch expects a file path, not a numpy array directly.
    We write the segment to a temporary WAV file, run prediction, then delete it.

    Returns a list of event dicts ready to be added to TranscriptionBuffer.
    """
    import tempfile, os, soundfile as sf
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    # Write segment to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_segment, SAMPLE_RATE)

    try:
        _, midi_data, _ = predict(
            tmp_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=args.onset_thresh,
            frame_threshold=args.frame_thresh,
            minimum_note_length=args.min_note_dur * 1000,   # ms
            midi_tempo=args.bpm,
        )
    finally:
        os.unlink(tmp_path)   # always clean up, even on error

    if midi_data is None or len(midi_data.instruments) == 0:
        return []

    # Collect all notes from the first (and only) instrument track
    raw_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            raw_notes.append({
                "start":      note.start,
                "end":        note.end,
                "midi":       note.pitch,
                "velocity":   note.velocity,
            })

    if not raw_notes:
        return []

    # Sort by start time
    raw_notes.sort(key=lambda n: n["start"])

    # Group into events: notes starting within CHORD_WINDOW of each other
    # are considered a chord.
    CHORD_WINDOW = 0.08   # seconds — notes within 80 ms = simultaneous

    events   = []
    group    = [raw_notes[0]]
    now_wall = time.time()   # approximate wall time for the event timestamp

    for note in raw_notes[1:]:
        if note["start"] - group[0]["start"] <= CHORD_WINDOW:
            group.append(note)
        else:
            events.append(_make_event(group, now_wall))
            group = [note]
    events.append(_make_event(group, now_wall))

    return events


def _make_event(note_group: list, wall_time: float) -> dict:
    """Convert a group of simultaneous notes into a transcription event dict."""
    note_dicts = []
    for n in note_group:
        midi  = int(n["midi"])
        name  = midi_to_note_name(midi)
        pc    = midi % 12
        note_dicts.append({
            "midi":        midi,
            "name":        name,
            "pitch_class": pc,
            "in_scale":    pc in scale_pc,
            "confidence":  round(float(n["velocity"]) / 127.0, 2),
        })

    duration_sec = float(np.mean([n["end"] - n["start"] for n in note_group]))
    rhythm       = quantize_duration(duration_sec, args.bpm)
    all_in_scale = all(n["in_scale"] for n in note_dicts)
    ev_type      = "chord" if len(note_dicts) > 1 else "note"

    return {
        "type":     ev_type,
        "notes":    note_dicts,
        "duration": round(duration_sec, 3),
        "rhythm":   rhythm,
        "in_scale": all_in_scale,
        "time":     wall_time,
    }


# ─────────────────────────────────────────────
# 8. MAIN REAL-TIME LOOP
# ─────────────────────────────────────────────

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """sounddevice callback: push raw audio into the processing queue."""
    if status:
        print(f"  [audio warning] {status}", flush=True)
    audio_queue.put(indata.copy())


def main():
    global RMS_ONSET_THRESHOLD, RMS_SILENCE_THRESHOLD

    # Lazy-import basic-pitch here so startup errors are clear
    try:
        from basic_pitch.inference import predict   # noqa: F401
        print("  basic-pitch loaded.")
    except ImportError:
        print("  ERROR: basic-pitch is not installed.")
        print("  Run: pip install basic-pitch")
        raise SystemExit(1)

    buffer           = TranscriptionBuffer()
    audio_accum      = []
    audio_accum_dur  = 0.0
    last_onset_time  = 0.0
    in_sound         = False
    silence_dur      = 0.0  # tracks how long RMS has been below silence threshold
    full_recording   = []   # accumulates ALL audio for WAV export
    sessions_dir = Path(__file__).parent / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_file = sessions_dir / f"session_{timestamp}.json"
    wav_file     = sessions_dir / f"session_{timestamp}.wav"
    active_device_idx  = args.device if args.device is not None else sd.default.device[0]
    active_device_name = sd.query_devices(active_device_idx)["name"]

    print("=" * 60)
    print("  Real-time Transcription — Piano")
    print(f"  Scale    : {args.scale} {args.type}  |  BPM : {args.bpm}")
    print(f"  Detector : basic-pitch (CNN polyphonic)")
    print(f"  Mic      : [{active_device_idx}] {active_device_name}")
    print(f"  Session  : {session_file}")
    print("  Press Ctrl+C to stop and save.")
    print("=" * 60)

    # Dynamic noise-floor calibration
    onset_thresh, silence_thresh = calibrate_noise_floor(args.device)
    RMS_ONSET_THRESHOLD   = onset_thresh
    RMS_SILENCE_THRESHOLD = silence_thresh
    print("  Ready — you may start playing.")
    print("-" * 60)

    with sd.InputStream(
        device=args.device,
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=HOP_SIZE,
    ):
        try:
            while True:
                data = audio_queue.get()
                mono = np.squeeze(data).astype(np.float32)
                now  = time.time()

                # Accumulate everything for the session WAV
                full_recording.append(mono)

                # ── RMS energy ────────────────────────────────────────────
                rms = float(np.sqrt(np.mean(mono ** 2)))

                # ── State transitions ─────────────────────────────────────
                hop_dur = HOP_SIZE / SAMPLE_RATE   # ~23ms per hop

                onset_detected = (rms > RMS_ONSET_THRESHOLD
                                  and (now - last_onset_time) > ONSET_COOLDOWN_SEC
                                  and not in_sound)

                # Silence must persist for SILENCE_HOLD_SEC before we react
                if in_sound:
                    if rms < RMS_SILENCE_THRESHOLD:
                        silence_dur += hop_dur
                    else:
                        silence_dur = 0.0
                silence_detected = (in_sound
                                    and silence_dur >= SILENCE_HOLD_SEC
                                    and audio_accum_dur >= MIN_SEGMENT_SEC)

                if onset_detected:
                    in_sound        = True
                    silence_dur     = 0.0
                    last_onset_time = now
                    audio_accum     = [mono]
                    audio_accum_dur = hop_dur

                elif in_sound:
                    audio_accum.append(mono)
                    audio_accum_dur += hop_dur

                # ── Trigger analysis ──────────────────────────────────────
                should_process = (
                    silence_detected
                    or (in_sound and audio_accum_dur >= MAX_SEGMENT_SEC)
                )

                if should_process and audio_accum:
                    in_sound      = False
                    audio_segment = np.concatenate(audio_accum)

                    # Run basic-pitch on the full segment
                    events = analyse_segment(audio_segment, audio_accum_dur)

                    for event in events:
                        buffer.add(event)
                        buffer.display_last()

                    audio_accum     = []
                    audio_accum_dur = 0.0

        except KeyboardInterrupt:
            import soundfile as sf

            # Save JSON transcript
            buffer.to_json(session_file)

            # Save full session WAV (for chord_analysis.py)
            if full_recording:
                wav_audio = np.concatenate(full_recording)
                sf.write(str(wav_file), wav_audio, SAMPLE_RATE)

            print("\n" + "=" * 60)
            print(f"  Session ended — {len(buffer.events)} events recorded.")
            print(f"  JSON  : {session_file}")
            print(f"  WAV   : {wav_file}")
            print(f"  → Run chord_analysis.py {session_file.name} for harmony analysis.")
            print("=" * 60)


if __name__ == "__main__":
    main()