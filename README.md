# MusicAgent
Tentative of developing an agent that could help any user in its learning and practice of music, more specifically piano.

The goal is to guide learners toward playing *expressively*, not only to correct wrong notes. A learner says how they want a piece to sound (for example *sad*); the system listens, estimates which emotion the performance actually conveys, and explains which playing choices (loudness, tempo, articulation, timing) would bring it closer to the intention. A chatbot turns this into coaching.

## How it is built

Only the last layer is original research. Everything it relies on uses widely used tools and textbook methods, so that the expressive feedback is the single thing being studied.

| Layer | What it does | How | Files |
|---|---|---|---|
| 1. Recognition | notes, key, chords, tempo | [basic-pitch](https://github.com/spotify/basic-pitch) (Spotify), [music21](https://web.mit.edu/music21/), [librosa](https://librosa.org) | `recognition.py`, `transcription.py`, `chord_analysis.py` |
| 2. Description | expressive descriptors: tempo, loudness, articulation, timing variation, register, mode… | standard descriptors computed with librosa and music21 | `expression_analysis.py` |
| 3. Recognition models | emotion of a performance; chords from audio, with their evaluations | **Random Forest** (scikit-learn) | `emotion_model.py`, `chord_rf.py`, `chord_evaluation.py` |
| 4. **Expressive feedback** | gap between intended and perceived emotion → prioritised advice | **this project's contribution** | `expressive_feedback.py` |
| Coaching | conversation with the learner | local LLM (Ollama) or Azure OpenAI | `agent.py` |

Supporting steps: correction feedback on notes, tempo and harmony (`feedback.py`), sheet music export (`json_to_score.py`), piano-roll images (`piano_roll.py`), figures for presentations and papers (`make_figures.py`), and `main.py`, which runs everything in one command.

Every design choice, what is out of scope and why, and the evidence behind each decision are in **[docs/decisions.md](docs/decisions.md)**.

### The emotion model

A Random Forest trained on [EMOPIA](https://zenodo.org/records/5257995) (1,078 pop piano clips labelled in the four quadrants of Russell's valence–arousal model: happy/excited, tense/angry, sad, calm/relaxed) from ten note-level descriptors. Evaluated by 5-fold cross-validation where each song stays in a single fold, and compared with standard classifiers on the same folds (mean ± standard deviation over folds):

| Model | 4 emotions | Macro-F1 | Arousal (energetic vs quiet) | Valence (positive vs negative) |
|---|---|---|---|---|
| Majority class (chance) | 28.8 % ± 2.8 | 0.11 | 52.5 % | 51.8 % |
| Logistic regression | 65.9 % ± 1.8 | 0.66 | 88.7 % | 73.6 % |
| k-NN (k = 15) | 64.7 % ± 3.3 | 0.64 | 89.9 % | 70.9 % |
| SVM (RBF kernel) | 67.2 % ± 2.3 | 0.67 | 89.6 % | 74.6 % |
| **Random Forest** | **67.9 % ± 3.1** | **0.68** | 88.7 % | 75.4 % |

All classic classifiers reach a similar level, and the Random Forest is at least as good as the others. The most important descriptors are loudness (MIDI velocity) and major/minor mode. Training writes the full report (tables, confusion matrix figure, feature importances) to `models/emotion_rf_report.md` and `models/emotion_confusion.png`.

### The chord models

Chords are named by **music21** from the transcribed notes (default). A **Random Forest** on beat-synchronous chroma (librosa), trained on synthetic piano chords (85 classes), is available as an alternative (`--chords rf`).

No chord-annotated piano recordings are freely available, so `chord_evaluation.py` tests both on (1) progressions in random keys and tempos, in three textures, rendered with sampled piano sounds (FluidSynth), and (2) a real piano recording of the 28 chords of [`protocols/chord_recording.md`](protocols/chord_recording.md). Scores are the standard MIREX chord metrics computed with mir_eval:

| Test | Recogniser | Root | Major/minor | Sevenths |
|---|---|---|---|---|
| Real piano, 28 isolated chords | **music21 (notes)** | 93 % | 100 % | **95 %** |
| | Random Forest (audio) | 100 % | 100 % | 45 % |
| Rendered progressions, 36 pieces | music21 (notes) | 52 % | 52 % | 36 % |
| | Random Forest (audio) | 92 % | 82 % | 70 % |

The two tests disagree. music21 is the default because the real recording is the most direct evidence, and there the Random Forest never recognises a seventh chord (a systematic bias from its synthetic training data). The full reasoning is in [docs/decisions.md](docs/decisions.md).

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

To re-run the chord evaluation, install FluidSynth and the soundfont (32 MB, free licence):

```bash
brew install fluid-synth
mkdir -p data/soundfonts
curl -L -o data/soundfonts/GeneralUser-GS.sf2 https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2
.venv311/bin/python chord_evaluation.py
.venv311/bin/python chord_evaluation.py --render-protocol expected.wav   # hear the 28-chord test
.venv311/bin/python chord_evaluation.py --recording take.wav             # score a real take
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

### Figures

```bash
.venv311/bin/python make_figures.py   # figures/*.png (slides), figures/*.pdf (paper), figures/captions.md
```

The figures are regenerated from the saved reports (model comparison, confusion matrix, feature importance, chord evaluations, an advice example, piano rolls), each with a caption that includes the current numbers.

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
- Chord recognition was tested on one real recording of isolated chords; music21 is weaker on chords played in context (rendered progressions).

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
- **mir_eval**: C. Raffel et al., "mir_eval: A Transparent Implementation of Common MIR Metrics", ISMIR 2014.
- **GeneralUser GS** soundfont by S. Christian Collins (free licence), rendered with FluidSynth.
- Emotion model: J. A. Russell, "A Circumplex Model of Affect", Journal of Personality and Social Psychology, 1980.
- Emotion cues: P. N. Juslin, P. Laukka, "Communication of Emotions in Vocal Expression and Music Performance", Psychological Bulletin, 2003.
