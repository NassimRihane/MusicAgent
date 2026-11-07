# MusicAgent
Tentative of developing an agent that could help any user in its learning and practice of music, more specifically piano.

The goal is to guide learners toward playing *expressively*, not only to correct wrong notes. A learner says how they want a piece to sound (for example *sad*); the system listens, estimates which emotion the performance actually conveys, and explains which playing choices (loudness, tempo, articulation, timing) would bring it closer to the intention. A chatbot turns this into coaching.

## How it is built

Only the last layer is original research. Everything it relies on uses widely used tools and textbook methods, so that the expressive feedback is the single thing being studied.

| Layer | What it does | How | Files |
|---|---|---|---|
| 1. Recognition | notes, key, chords, tempo | [basic-pitch](https://github.com/spotify/basic-pitch) (Spotify), [music21](https://web.mit.edu/music21/), [librosa](https://librosa.org) | `recognition.py`, `transcription.py`, `chord_analysis.py` |
| 2. Description | expressive descriptors: tempo, loudness, articulation, timing variation, register, mode… | standard descriptors computed with librosa and music21 | `expression_analysis.py` |
| 3. Recognition models | emotion of a performance; chords from audio | **Random Forest** (scikit-learn) | `emotion_model.py`, `chord_rf.py` |
| 4. **Expressive feedback** | gap between intended and perceived emotion → prioritised advice | **this project's contribution** | `expressive_feedback.py` |
| Coaching | conversation with the learner | local LLM (Ollama) or Azure OpenAI | `agent.py` |

Supporting steps: correction feedback on notes, tempo and harmony (`feedback.py`), sheet music export (`json_to_score.py`), piano-roll images (`piano_roll.py`), and `main.py`, which runs everything in one command.

### The emotion model

A Random Forest trained on [EMOPIA](https://zenodo.org/records/5257995) (1,078 pop piano clips labelled in four emotion quadrants) from ten note-level descriptors. Evaluated by 5-fold cross-validation where each song stays in a single fold:

| Task | Accuracy | Chance |
|---|---|---|
| 4 emotions (happy / tense / sad / calm) | 67.9 % | 28.9 % |
| Arousal (energetic vs quiet) | 88.7 % | 50 % |
| Valence (positive vs negative) | 75.4 % | 50 % |

The most important descriptors are loudness (MIDI velocity) and major/minor mode. Full report: `models/emotion_rf_report.json` (created by training).

### The expressive feedback (research part)

For a target emotion, `expressive_feedback.py`:

1. predicts which emotion the performance conveys;
2. measures, for each cue the learner controls (tempo, loudness, dynamic contrast, articulation, timing freedom), the distance to typical EMOPIA performances of the target emotion;
3. ranks these gaps by the model's reliance on each cue, and turns the top ones into advice;
4. re-runs the model with those cues changed, to check that the advice would actually move the performance toward the target.

Cues fixed by the piece (key, register, texture) are reported but never given as advice.

## Setup

Requires Python 3.11 (basic-pitch does not support newer versions on macOS) and [Ollama](https://ollama.com) for the chatbot.

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
ollama pull mistral

# Train the models once (downloads EMOPIA, 32 MB; about 4 minutes in total)
.venv311/bin/python emotion_model.py --train
.venv311/bin/python chord_rf.py --train
```

## Usage

```bash
# Record (Ctrl+C to stop), then get coaching toward an intended emotion
.venv311/bin/python main.py --emotion sad

# Analyse an existing recording instead
.venv311/bin/python main.py --audio "Happy Birthday.mp3" --emotion sad

# Analyse a previous session, in French, with another model
.venv311/bin/python main.py --session sessions/session_X.json --emotion calm --lang fr --model llama3.1
```

Emotions: happy, excited, tense, angry, sad, melancholic, calm, relaxed, tender… The key and tempo are detected automatically; to declare them, add `--scale G --type minor --bpm 80`. The chatbot's default model is set at the top of `agent.py` (`PROVIDER`, `MODEL`); `--dry-run` prints its prompt without calling any model.

### Individual steps

```bash
.venv311/bin/python transcription.py --audio take.mp3               # session JSON + WAV
.venv311/bin/python chord_analysis.py sessions/session_X.json --backend both   # music21 vs Random Forest
.venv311/bin/python expressive_feedback.py sessions/session_X.json --emotion sad
.venv311/bin/python emotion_model.py --predict take.mp3

# Expressive cues of two takes of the same piece, checked against music-emotion research
.venv311/bin/python expression_analysis.py happy.mp3 --compare sad.mp3 --same-piece

# Piano rolls of any recordings, as an image
.venv311/bin/python piano_roll.py happy.mp3 sad.mp3 --labels Happy Sad --title "Happy Birthday"
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| Recording + transcription | `transcription.py` | `session_X.json`, `session_X.wav` |
| Chord analysis | `chord_analysis.py` | `session_X_chords.json` |
| Sheet music (optional) | `json_to_score.py` | `.musicxml` / `.pdf` / `.mid` |
| Correction feedback | `feedback.py` | `feedback_session_X.json` |
| Expressive feedback | `expressive_feedback.py` | `expressive_session_X.json` |
| Chatbot | `agent.py` | `conversation_feedback_session_X.json` |

Early experiments are in `prototypes/`.

## Limitations

- EMOPIA is pop piano transcribed from YouTube recordings. Its notes include the sustain pedal and come from a different transcription system than basic-pitch, so typical values (articulation especially) are not directly comparable with our transcriptions.
- Without the score of the piece, rhythm can only be judged as *stable or not*, not *correct*: an intentional rubato and an unsteady pulse look the same.
- Chords from the audio Random Forest agree with music21's on the scale degree about 60–70 % of the time on real piano; it often confuses major and minor.

## Tests

```bash
.venv311/bin/pip install -r requirements-dev.txt
.venv311/bin/python -m pytest                 # all tests (~20 s)
.venv311/bin/python -m pytest -m "not slow"   # skip real-model / full-pipeline tests
```

Tests use synthesised audio, miniature datasets and fake LLM responses: they need no microphone, no downloaded data and make no network calls.

## Credits

- **EMOPIA**: H.-T. Hung, J. Ching, S. Doh, N. Kim, J. Nam, Y.-H. Yang, "EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation", ISMIR 2021. CC-BY 4.0.
- **basic-pitch**: R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal, S. Ewert, "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022.
- **music21**: M. S. Cuthbert, C. Ariza, "music21: A Toolkit for Computer-Aided Musicology and Symbolic Music Data", ISMIR 2010.
- **librosa**: B. McFee et al., "librosa: Audio and Music Signal Analysis in Python", SciPy 2015.
- **scikit-learn**: F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011.
- Emotion cues: P. N. Juslin, P. Laukka, "Communication of Emotions in Vocal Expression and Music Performance", Psychological Bulletin, 2003.
