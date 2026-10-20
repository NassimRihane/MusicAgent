# note_detection.py

import sounddevice as sd
import numpy as np
import aubio
import math
import queue
import time


# Select the desired scale

SELECTED_ROOT = "C"        # change if needed
SELECTED_SCALE_TYPE = "major"  # or "minor"

# Here "minor" assumes "natural minor"


# Parameters
A4_FREQ = 440.0
A4_MIDI = 69

sample_rate = 44100
buffer_size = 2048   # samples per frame
hop_size = 512       # hop between analyses
TOLERANCE_CENTS = 50  # tolerance to consider the note as that MIDI integer


# Scale (adaptable)

note_names_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_NAME_TO_PC = {
    'C': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#':3, 'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11 
}


SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10]
}

def build_scale(root, scale_type):
    """
    Generate the scale based on the parameters:
        - root: root of the scale wanted (C, G, ...)
        - type: "major" or "minor"
    """
    intervals = SCALE_INTERVALS[scale_type]
    return set((root + i) % 12 for i in intervals)



scale_pc = build_scale(
    NOTE_NAME_TO_PC[SELECTED_ROOT],
    SELECTED_SCALE_TYPE
)


# Converters

def freq_to_midi(f):
    """Convert frequency (Hz) to nearest MIDI note (float)."""
    return 69 + 12 * np.log2(f / A4_FREQ)

def midi_to_note_name(midi):
    """Round midi to nearest int and return note name with octave."""
    m_round = int(np.round(midi))
    pitch_class = m_round % 12
    octave = m_round // 12 - 1
    return f"{note_names_sharp[pitch_class]}{octave}"

def freq_to_note_name(f):
    midi = freq_to_midi(f)
    return midi_to_note_name(midi)

def note_name_to_pc(note_name):
    """Get pitch-class index 0..11 from a name like C#4 or Db3.
       Accepts flats 'b' as well (Db)."""
    # Normalize flats to sharps:
    name = note_name.strip()
    # remove octave digits
    base = ''.join(ch for ch in name if not ch.isdigit())
    # handle flats
    base = base.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#')
    base = base.replace('Ab', 'G#').replace('Bb', 'A#')
    if base in note_names_sharp:
        return note_names_sharp.index(base)
    raise ValueError("Unknown note name: " + note_name)

def freq_to_note_info(f):
    """
    Convert frequency to:
    - MIDI note (int)
    - note name with octave (str)
    - cents deviation (float)
    - pitch class (int 0..11)
    """
    midi_float = freq_to_midi(f)
    midi_round = int(np.round(midi_float))

    cents = 100 * (midi_float - midi_round)

    pitch_class = midi_round % 12
    octave = midi_round // 12 - 1
    note_name = f"{note_names_sharp[pitch_class]}{octave}"

    return midi_round, note_name, cents, pitch_class



# Aubio pitch detector

pitch_o = aubio.pitch("yin", buffer_size, hop_size, sample_rate)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)  # dB

# Thread-safe queue for audio
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """sounddevice callback: put audio into queue for processing"""
    if status:
        print(status, flush=True)
    q.put(indata.copy())



def is_in_scale(pc_set, pitch_class):
    return pitch_class % 12 in pc_set

print("Starting real-time pitch detection. Press Ctrl+C to stop.")
with sd.InputStream(channels=1,
                    callback=audio_callback,
                    samplerate=sample_rate,
                    blocksize=hop_size):
    try:
        while True:
            data = q.get()
            mono = np.squeeze(data)

            samples = mono.astype(np.float32)
            pitch = pitch_o(samples)[0]
            confidence = pitch_o.get_confidence()

            if pitch > 0 and confidence > 0.3:
                m, name, cents, pc = freq_to_note_info(pitch)
                in_scale = pc in scale_pc
                status = "IN SCALE" if in_scale else "OUTSIDE SCALE"
                print(
                    f"{name} | {pitch:.1f} Hz | "
                    f"{cents:+.1f} cents | {status}"
                )

    except KeyboardInterrupt:
        print("Stopped.")
