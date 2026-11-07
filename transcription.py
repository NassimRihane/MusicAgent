# transcription.py
#
# Step 1 — Recording and transcription
#
# Records the piano from the microphone (or takes an existing audio file),
# then transcribes the WHOLE recording with basic-pitch, detects the key with
# music21 and the tempo with librosa (see recognition.py), and saves a
# session JSON next to the session WAV.
#
# While recording, a live preview prints the notes as they are played
# (basic-pitch on short segments). The preview is only for display; the
# session is always built from the full recording, as basic-pitch is meant
# to be used.
#
# Usage:
#   python transcription.py                                   # record; key and tempo detected
#   python transcription.py --scale G --type minor --bpm 80   # declare the intended key / tempo
#   python transcription.py --audio "Happy Birthday.mp3"      # analyse an existing recording
#   python transcription.py --list-devices

import argparse
import queue
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa

import recognition as rc


# ─────────────────────────────────────────────
# 0. COMMAND-LINE ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Piano recording and transcription")
    parser.add_argument("--audio",        default=None,
                        help="Analyse this audio file instead of recording")
    parser.add_argument("--scale",        default=None,
                        help="Intended key's tonic (e.g. C, G, F#). Default: detected by music21")
    parser.add_argument("--type",         default="major",
                        choices=["major", "minor"], help="Intended key's mode (with --scale)")
    parser.add_argument("--bpm",          type=float, default=None,
                        help="Intended tempo. Default: estimated by librosa")
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
    return parser.parse_args(argv)


def list_devices():
    print("\nAvailable audio input devices:\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marker = " ◄ (default)" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}]  {d['name']}{marker}")
    print("\nUse --device <index> to select a microphone.")


# ─────────────────────────────────────────────
# 1. CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────

SAMPLE_RATE  = rc.SAMPLE_RATE   # basic-pitch works natively at 22050 Hz
HOP_SIZE     = 512              # audio callback block size (~23 ms per hop)
SESSIONS_DIR = Path(__file__).parent / "sessions"

# Live preview: onset / silence detection (RMS-based, calibrated at startup)
RMS_ONSET_THRESHOLD   = 0.02   # overwritten after calibration
RMS_SILENCE_THRESHOLD = 0.005  # overwritten after calibration
ONSET_COOLDOWN_SEC    = 0.15   # minimum gap between two onsets

# Live preview segments: collected until silence, then sent to basic-pitch.
# MIN_SEGMENT_SEC: segment must be at least this long before analysis.
# MAX_SEGMENT_SEC: hard cap — send for analysis even if no silence detected.
# SILENCE_HOLD_SEC: silence must persist this long before a segment ends.
MIN_SEGMENT_SEC  = 0.4
MAX_SEGMENT_SEC  = 4.0
SILENCE_HOLD_SEC = 0.15


# ─────────────────────────────────────────────
# 2. SESSION BUILDING (full-recording analysis)
# ─────────────────────────────────────────────

def build_session(wav_path: Path, declared_key: str = None, declared_bpm: float = None,
                  source: str = None, onset_thresh: float = 0.5, frame_thresh: float = 0.3,
                  min_note_dur: float = 0.05) -> Path:
    """
    Transcribe a session WAV with basic-pitch, detect key (music21) and tempo
    (librosa), and write session_X.json next to it. Declared key / tempo, when
    given, are used for the session; detected ones are always recorded too.
    """
    wav_path = Path(wav_path)
    print("  Transcribing the full recording with basic-pitch...")
    notes = rc.transcribe(wav_path, onset_thresh, frame_thresh, min_note_dur)
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

    detected = rc.detect_key(notes)
    beats = rc.track_beats(audio=audio)
    key_name = declared_key or detected["key"]
    bpm = declared_bpm or beats["tempo_bpm"]
    scale_pc = rc.scale_pitch_classes(key_name) if key_name else None

    session = {
        "meta": {
            "scale":           key_name,
            "scale_source":    "declared" if declared_key else "detected (music21)",
            "key_detected":    detected["key"],
            "key_correlation": detected["correlation"],
            "bpm":             bpm,
            "bpm_source":      "declared" if declared_bpm else "estimated (librosa)",
            "tempo_estimated": beats["tempo_bpm"],
            "date":            datetime.now().isoformat(),
            "audio":           wav_path.name,
            "source":          source,
            "transcription":   "basic-pitch (full recording)",
        },
        "events": rc.notes_to_events(notes, scale_pc, bpm),
        "notes":  [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in n.items()}
                   for n in notes],
    }
    out = wav_path.with_suffix(".json")
    out.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {len(notes)} notes  |  key {key_name} ({session['meta']['scale_source']})"
          f"  |  {bpm:.0f} BPM ({session['meta']['bpm_source']})")
    return out


def import_audio(src, wav_path: Path) -> Path:
    """Convert any audio file librosa can read to a mono 22050 Hz session WAV."""
    import soundfile as sf
    audio, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    return wav_path


# ─────────────────────────────────────────────
# 3. LIVE PREVIEW
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

    rms_values = [float(np.sqrt(np.mean(audio[i:i + HOP_SIZE] ** 2)))
                  for i in range(0, len(audio) - HOP_SIZE, HOP_SIZE)]
    noise_rms = float(np.percentile(rms_values, 95))

    # ×10 for onset: the signal must be 10× louder than the background noise.
    # ×3 for silence: a note is over when RMS drops to 3× the noise floor.
    # A hard floor keeps the thresholds meaningful in a very quiet room.
    noise_rms_floored = max(noise_rms, 0.001)
    onset_threshold   = noise_rms_floored * 10.0
    silence_threshold = noise_rms_floored * 3.0

    print(f"  Noise floor: RMS={noise_rms:.5f} (floored={noise_rms_floored:.5f}) → "
          f"onset={onset_threshold:.5f} / silence={silence_threshold:.5f}")
    return onset_threshold, silence_threshold


def analyse_segment(audio_segment: np.ndarray, segment_start: float,
                    scale_pc: set = None, bpm: float = 60.0) -> list:
    """
    Live preview: basic-pitch on one segment (written to a temporary WAV),
    returned as session events with times shifted by segment_start.
    """
    import tempfile, os, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_segment, SAMPLE_RATE)
    try:
        notes = rc.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)   # always clean up, even on error
    for n in notes:
        n["start"] += segment_start
        n["end"] += segment_start
    return rc.notes_to_events(notes, scale_pc, bpm)


def display_event(e: dict):
    """Print one preview event."""
    names = " + ".join(n["name"] for n in e["notes"])
    kind = "CHORD" if len(e["notes"]) > 1 else "NOTE "
    scale = "" if e["in_scale"] is None else ("in scale ✓" if e["in_scale"] else "out of scale ✗")
    print(f"[{e['time']:6.2f}s]  {kind}  {names:20s} | {e['rhythm']:16s} | {scale}")


audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """sounddevice callback: push raw audio into the processing queue."""
    if status:
        print(f"  [audio warning] {status}", flush=True)
    audio_queue.put(indata.copy())


def record(wav_path: Path, device=None, scale_pc: set = None, bpm: float = None):
    """
    Record until Ctrl+C with a live note preview, then save the WAV.
    Returns the WAV path, or None if nothing was recorded.
    """
    global RMS_ONSET_THRESHOLD, RMS_SILENCE_THRESHOLD
    preview_bpm      = bpm or 60.0
    audio_accum      = []
    audio_accum_dur  = 0.0
    last_onset_time  = 0.0
    in_sound         = False
    silence_dur      = 0.0  # how long RMS has been below the silence threshold
    full_recording   = []   # accumulates ALL audio for the session WAV
    samples_seen     = 0    # audio clock: position in the session WAV
    segment_start    = 0.0  # session time (s) of the current segment's first block

    RMS_ONSET_THRESHOLD, RMS_SILENCE_THRESHOLD = calibrate_noise_floor(device)
    print("  Ready — you may start playing. Press Ctrl+C to stop.")
    print("-" * 60)

    with sd.InputStream(device=device, channels=1, callback=audio_callback,
                        samplerate=SAMPLE_RATE, blocksize=HOP_SIZE):
        try:
            while True:
                mono = np.squeeze(audio_queue.get()).astype(np.float32)
                # Audio clock, not wall time: analysis blocks this loop, so blocks
                # are read late but their position in the stream is exact.
                block_start   = samples_seen / SAMPLE_RATE
                samples_seen += len(mono)
                full_recording.append(mono)

                rms     = float(np.sqrt(np.mean(mono ** 2)))
                hop_dur = HOP_SIZE / SAMPLE_RATE
                onset_detected = (rms > RMS_ONSET_THRESHOLD
                                  and (block_start - last_onset_time) > ONSET_COOLDOWN_SEC
                                  and not in_sound)
                if in_sound:
                    silence_dur = silence_dur + hop_dur if rms < RMS_SILENCE_THRESHOLD else 0.0
                silence_detected = (in_sound and silence_dur >= SILENCE_HOLD_SEC
                                    and audio_accum_dur >= MIN_SEGMENT_SEC)

                if onset_detected:
                    in_sound, silence_dur = True, 0.0
                    last_onset_time = segment_start = block_start
                    audio_accum, audio_accum_dur = [mono], hop_dur
                elif in_sound:
                    audio_accum.append(mono)
                    audio_accum_dur += hop_dur

                if (silence_detected or (in_sound and audio_accum_dur >= MAX_SEGMENT_SEC)) \
                        and audio_accum:
                    in_sound = False
                    for event in analyse_segment(np.concatenate(audio_accum), segment_start,
                                                 scale_pc, preview_bpm):
                        display_event(event)
                    audio_accum, audio_accum_dur = [], 0.0

        except KeyboardInterrupt:
            pass

    if not full_recording:
        return None
    import soundfile as sf
    sf.write(str(wav_path), np.concatenate(full_recording), SAMPLE_RATE)
    print("\n" + "-" * 60)
    print(f"  Recording saved: {wav_path}")
    return wav_path


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return

    declared_key = f"{args.scale} {args.type}" if args.scale else None
    SESSIONS_DIR.mkdir(exist_ok=True)
    wav_path = SESSIONS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    print("=" * 60)
    print("  Piano transcription — basic-pitch")
    print(f"  Key   : {declared_key or 'detected after recording (music21)'}")
    print(f"  Tempo : {args.bpm or 'estimated after recording (librosa)'}")
    if args.audio:
        print(f"  Input : {args.audio}")
        print("=" * 60)
        import_audio(args.audio, wav_path)
        source = Path(args.audio).name
    else:
        device = args.device if args.device is not None else sd.default.device[0]
        print(f"  Mic   : [{device}] {sd.query_devices(device)['name']}")
        print("=" * 60)
        scale_pc = rc.scale_pitch_classes(declared_key) if declared_key else None
        if record(wav_path, args.device, scale_pc, args.bpm) is None:
            print("  Nothing was recorded.")
            return
        source = "microphone"

    session = build_session(wav_path, declared_key, args.bpm, source,
                            args.onset_thresh, args.frame_thresh, args.min_note_dur)
    print("=" * 60)
    print(f"  JSON : {session}")
    print(f"  WAV  : {wav_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
