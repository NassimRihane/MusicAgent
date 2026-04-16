# chord_detection.py

import sounddevice as sd
import numpy as np
import librosa
import time


# Parameters
SR = 44100
DURATION = 1.5   # seconds to listen for a chord
N_FFT = 4096
HOP_LENGTH = 512


# Pitch classes
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# Chord templates (C major only)
CHORD_TEMPLATES = {
    "C":     [1,0,0,0,1,0,0,1,0,0,0,0],  # do mi sol
    "Dm":    [0,0,1,0,0,1,0,0,0,1,0,0],  # re fa la
    "Em":    [0,0,0,0,1,0,0,1,0,0,0,1],  # mi sol si
    "F":     [1,0,0,0,0,1,0,0,0,1,0,0],  # fa la do
    "G":     [0,0,1,0,0,0,0,1,0,0,0,1],  # sol si re
    "Am":    [1,0,0,0,1,0,0,0,0,1,0,0],  # la do mi
    "Bdim":  [0,0,1,0,0,1,0,0,0,0,0,1],  # si re fa
}

# Normalize templates
for chord in CHORD_TEMPLATES:
    CHORD_TEMPLATES[chord] = np.array(CHORD_TEMPLATES[chord], dtype=float)
    CHORD_TEMPLATES[chord] /= np.linalg.norm(CHORD_TEMPLATES[chord])



# Record audio
def record_audio(duration):
    print("Play a chord now...")
    audio = sd.rec(int(SR * duration), samplerate=SR,
                   channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


# Compute chroma vector
def compute_chroma(y):
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean /= (np.linalg.norm(chroma_mean) + 1e-6)
    return chroma_mean


# Recognize chord
def recognize_chord(chroma_vec):
    best_chord = None
    best_score = -1

    for chord_name, template in CHORD_TEMPLATES.items():
        score = np.dot(chroma_vec, template)
        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord, best_score


# Main loop
if __name__ == "__main__":
    while True:
        y = record_audio(DURATION)
        chroma_vec = compute_chroma(y)
        chord, confidence = recognize_chord(chroma_vec)

        print(f"Detected chord: {chord} (confidence={confidence:.2f})")
        print("-" * 40)
        time.sleep(0.5)
