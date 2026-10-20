# MusicAgent
Tentative of developing an agent that could help any user in its learning and practice of music, more specifically piano.

The goal is to guide learners toward playing *expressively* (e.g. playing a piece happily), not only to correct wrong notes.

## Setup

Requires Python 3.11 (basic-pitch does not support newer versions on macOS) and [Ollama](https://ollama.com) for the chatbot.

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
ollama pull mistral
```

## Usage

```bash
# Record (Ctrl+C to stop), analyse, then chat about the session
.venv311/bin/python main.py --scale C --type major --bpm 60

# Analyse an existing session
.venv311/bin/python main.py --session sessions/session_20260101_120000.json

# Another model or language
.venv311/bin/python main.py --session sessions/session_X.json --model llama3.1 --lang fr
```

The default model is set at the top of `agent.py` (`PROVIDER`, `MODEL`).

## Pipeline

| Step | Script | Output |
|---|---|---|
| Recording + transcription | `transcription.py` | `session_X.json`, `session_X.wav` |
| Chord analysis | `chord_analysis.py` | `session_X_chords.json` |
| Sheet music (optional) | `json_to_score.py` | `.musicxml` / `.pdf` / `.mid` |
| Feedback | `feedback.py` | `feedback_session_X.json` |
| Chatbot | `agent.py` | `conversation_feedback_session_X.json` |

Early experiments are in `prototypes/`.
