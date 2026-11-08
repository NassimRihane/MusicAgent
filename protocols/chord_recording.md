# Chord recording test

This recording measures how well the chord recognisers work on a real piano.
The answers are known in advance, so `chord_evaluation.py --recording` can score them.

## How to record

- Use the piano and microphone you normally practise with.
- Play the 28 chords **in order, in one take**.
- For each chord: strike it once with **both hands together**, hold it for about **2 seconds**,
  release it, leave about **1 second of silence**, then play the next one.
- Use no sustain pedal, so that each chord stops when you release it.
- If you make a mistake, stop and start the take again from chord 1.
- Recording two takes is even better (for example on two different days).

To record with the project: `python transcription.py` (Ctrl+C to stop), then
`python chord_evaluation.py --recording sessions/session_XXXXXXXX_XXXXXX.wav`.
Any audio file works (wav, mp3, m4a from a phone...).

## The chords

| # | Chord | Type | Left hand | Right hand |
|---|---|---|---|---|
| 1 | **C** | major | C2 | C4 E4 G4 |
| 2 | **Am** | minor | A1 | A3 C4 E4 |
| 3 | **F** | major | F2 | F4 A4 C5 |
| 4 | **G7** | dominant 7th | G2 | G4 B4 D5 F5 |
| 5 | **Dm** | minor | D2 | D4 F4 A4 |
| 6 | **E** | major | E2 | E4 G#4 B4 |
| 7 | **Bdim** | diminished | B1 | B3 D4 F4 |
| 8 | **Cmaj7** | major 7th | C2 | C4 E4 G4 B4 |
| 9 | **F#m** | minor | F#2 | F#4 A4 C#5 |
| 10 | **Bb** | major | Bb1 | Bb3 D4 F4 |
| 11 | **Daug** | augmented | D2 | D4 F#4 A#4 |
| 12 | **D7** | dominant 7th | D2 | D4 F#4 A4 C5 |
| 13 | **Gm7** | minor 7th | G2 | G4 Bb4 D5 F5 |
| 14 | **Ab** | major | Ab1 | Ab4 C5 Eb5 |
| 15 | **C#dim** | diminished | C#2 | C#4 E4 G4 |
| 16 | **Ebmaj7** | major 7th | Eb2 | Eb4 G4 Bb4 D5 |
| 17 | **Bm** | minor | B1 | B3 D4 F#4 |
| 18 | **A7** | dominant 7th | A1 | A3 C#4 E4 G4 |
| 19 | **Fm7** | minor 7th | F2 | F4 Ab4 C5 Eb5 |
| 20 | **Caug** | augmented | C2 | C4 E4 G#4 |
| 21 | **Gmaj7** | major 7th | G2 | G4 B4 D5 F#5 |
| 22 | **C#m** | minor | C#2 | C#4 E4 G#4 |
| 23 | **F#dim** | diminished | F#2 | F#4 A4 C5 |
| 24 | **E7** | dominant 7th | E2 | E4 G#4 B4 D5 |
| 25 | **Em7** | minor 7th | E2 | E4 G4 B4 D5 |
| 26 | **Dmaj7** | major 7th | D2 | D4 F#4 A4 C#5 |
| 27 | **G#dim** | diminished | G#1 | G#4 B4 D5 |
| 28 | **Cm7** | minor 7th | C2 | C4 Eb4 G4 Bb4 |

Middle C is C4. Left hand: the root, low. Right hand: the chord in root position.

Note: augmented chords are symmetric (C, E and G# augmented contain the same notes), so no recogniser can tell their root from the sound alone.
