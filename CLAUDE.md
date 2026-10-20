# CLAUDE.md

Context for Claude Code sessions on this project. Read it first; update the
**Change log** and **Known issues** sections whenever you change the code.

## Project goal

A research prototype (Keio) for an AI assistant teacher for music learners
(piano). The goal is to **guide the learner toward playing expressively**,
not only to correct mistakes. Example: the learner wants to play a piece
*happily*. The system should know what makes a performance sound happy
(tempo, articulation, dynamics, mode…), find where the player falls short,
and give personalised feedback through a chatbot.

Current state: the pipeline only measures note accuracy against a key,
rhythmic regularity and chord fit. **No expressive analysis exists yet.**
That is the main roadmap item (see below).

## Ground rules

- **Never call Azure OpenAI.** The user has no Azure credits left. The
  default provider is local **Ollama** (`mistral` installed), which is free
  and fine to call for tests. `--dry-run` prints the prompt and makes no call.
- **The GitHub repo is public** (`NassimRihane/MusicAgent`). Never commit
  `sessions/` (the user's recordings; it's gitignored), credentials or `.env`
  files. Commit or push only when the user asks; never force-push.
- **Don't overwrite files in `sessions/`.** They are the user's real
  recordings. Run tests on copies in a scratch directory (`feedback.py` and
  `chord_analysis.py` write their outputs next to their input).
- Match the existing style: section banners (`# ───── N. TITLE ─────`),
  aligned assignments, a module header comment with usage. Each script has
  `parse_args(argv=None)` in section 0 and `main(argv=None)`; **never read
  CLI args at import time**, and pass values to functions as parameters.
- **Run the tests after any code change:** `.venv311/bin/python -m pytest`
  (~8 s), or `-m "not slow"` to skip real basic-pitch and the pipeline
  subprocess. Add or update tests along with code changes.
- The user writes in English and sometimes French; the agent supports
  `--lang en|fr`.

## Environment

- Run everything with **`.venv311/bin/python`** (Python 3.11 from Homebrew,
  `/opt/homebrew/bin/python3.11`). Rebuilt on 2026-10-20 from
  `requirements.txt`; verified that basic-pitch loads and transcribes.
- **Python 3.11 is required.** On macOS, basic-pitch 0.4.0 uses CoreML only
  on 3.11; on 3.12+ it requires `tensorflow-macos<2.15.1`, which doesn't exist.
- `requirements.txt` pins `setuptools<81` (resampy imports `pkg_resources`)
  and `chardet<6` (otherwise requests warns).
- Ollama runs locally at `http://localhost:11434` (version 0.34.2); installed
  model: `mistral`.
- Git: `main` tracks `origin` = https://github.com/NassimRihane/MusicAgent
  (public). `core.fileMode=false`, because every file on disk is mode 755.
  No `gh` CLI is installed.

## Tests

`tests/`, one file per module; config in `pytest.ini`; `pytest` is in
`requirements-dev.txt`.

- `conftest.py` blocks `requests.post` in every test (no Ollama or Azure
  calls; agent tests install fakes that record the payload). It also
  provides synthetic audio helpers (`tones`, `silence`) and session builders
  (`event`, `make_session`, `write_json`). Tests write only to `tmp_path`.
- `slow` marker: real basic-pitch on synthesised notes, and the full
  `main.py --session … --dry-run` pipeline as subprocesses.
- One **strict xfail** documents the rhythm-metric limitation (see Known
  issues). When the metric is redesigned, it will XPASS and fail the run:
  remove the xfail marker then.
- Mutation-checked on 2026-10-20: reintroducing old bugs (`self.h`, the
  `assessement` key, wall-clock timestamps, no skipping of simultaneous
  onsets) makes tests fail.

## Pipeline

`main.py` runs every step in order (or `--session X.json` to skip
recording); each step is still runnable on its own:

```
transcription.py   mic → session_X.json (note events) + session_X.wav
        │
        ├─ json_to_score.py   session_X.json → .musicxml / .pdf (LilyPond) / .mid
        ├─ chord_analysis.py  session_X.wav  → session_X_chords.json/.txt
        │                     (uses chord_model.py if trained weights exist)
        ▼
feedback.py        session_X.json (+ _chords.json, auto-detected)
                   → feedback_session_X.json/.txt  (notes / rhythm / harmony + suggestions)
        ▼
agent.py           feedback_*.json → chat loop (Ollama or Azure) → conversation_feedback_*.json
```

Model selection: the `PROVIDER` / `MODEL` constants at the top of `agent.py`,
or `--provider ollama|azure` and `--model <name>` (also on `main.py`). Both
providers use the OpenAI chat format (Ollama via `/v1/chat/completions`).

Legacy prototypes in `prototypes/`, not part of the pipeline: `note_detection.py` (aubio,
real-time pitch, C major hard-coded), `chord_detection.py` (live chord
templates, C major only), `sim_proba.py` (a Bayesian simulation of learner
motivation that could later feed a learner model).

### Session JSON schema (transcription.py output)

```json
{"meta": {"scale": "C major", "bpm": 60.0, "date": "..."},
 "events": [{"type": "note|chord", "time": 12.12, "duration": 0.5,
             "rhythm": "eighth", "in_scale": true,
             "notes": [{"midi": 60, "name": "C4", "pitch_class": 0,
                        "in_scale": true, "velocity": 90}]}]}
```

- `time` is the onset in seconds from stream start, **aligned with the
  session WAV** (since the 2026-10-10 change).
- Rhythm labels: `whole, half, quarter, eighth, 16th, dotted-half,
  dotted-quarter, dotted-eighth, quarter-triplet, eighth-triplet`.
- Old sessions (before 2026-10-10) store `confidence` (= velocity/127)
  instead of `velocity`, and have **broken `time` values** (see Known
  issues). Readers must accept both formats.

## Change log

### 2026-10-20 (b): Automated tests, argparse refactor

- **Refactor:** module-level `parser`/`args` → `parse_args(argv=None)` +
  `main(argv=None)` in all 6 scripts, so modules can be imported.
  Signatures changed:
  - transcription: `TranscriptionBuffer(scale, bpm)`,
    `analyse_segment(audio, segment_start, scale_pc, bpm, onset_thresh,
    frame_thresh, min_note_dur)`,
    `_make_event(group, segment_start, scale_pc, bpm)`; `scale_pc` is no
    longer a module global; `--list-devices` → `list_devices()`
  - chord_analysis: `analyse_librosa(audio, segment_sec, hop_sec)`,
    `run_analysis(audio, backend, segment_sec, hop_sec)`
  - agent: `load_config(provider, model, max_tokens)` (`max_tokens` is
    stored in the config), `load_ollama_config(model)`
  CLI usage is unchanged (checked with `--help` and `--list-devices`).
- **Tests:** 84 tests + 1 xfail across 6 files (see the Tests section).
  Added `pytest.ini` and `requirements-dev.txt`.
- `.gitignore`: the user added `.venv311*/` and `venv311/`; this is
  redundant with `.venv*/` but harmless. The "10k changes" seen in VSCode
  were a stale view from before the venv rules existed.

### 2026-10-20: Git, venv, Ollama, main.py

- **Git:** the folder is now a clone of `NassimRihane/MusicAgent`. It was
  attached with `git fetch` + `git reset origin/main` (working tree
  untouched), and the local code was committed on top of the 3 existing
  GitHub commits (no force-push). Before that, I checked that GitHub held no
  code missing locally. `.gitignore` extended (`__pycache__`, `.venv*`,
  `.DS_Store`, `.claude/settings.local.json`; `sessions/` was already there).
- **venv:** `.venv311` rebuilt with Homebrew Python 3.11.17;
  `requirements.txt` gained the `setuptools<81` and `chardet<6` pins.
- **Prototypes** moved to `prototypes/` (`note_detection.py`,
  `chord_detection.py`, `sim_proba.py`); nothing imported them.
- **agent.py:** Ollama support (now the default). `PROVIDER` / `MODEL`
  constants plus `--provider` / `--model`. `load_config()` returns
  `{provider, url, headers, model, timeout}` (180 s timeout for Ollama, 30 s
  for Azure). Connection and 404 errors give Ollama-specific hints. The
  conversation log stores `provider` and `model` instead of `deployment`.
  Tested with a real `mistral` conversation.
- **main.py (new):** runs transcription → chord_analysis → (json_to_score
  with `--score`) → feedback → agent as subprocesses. During a step, Ctrl+C
  goes to the child (it stops the recording) while main.py keeps waiting.
  After recording, it picks the newest `sessions/session_*.json`. Tested with
  `--session` on a copy; **the recording path is untested** (needs the
  microphone).

### 2026-10-10: Full review and bug-fix pass

**transcription.py**
- Timestamps fixed. Previously every event from one audio segment got
  `time.time()` taken *after* basic-pitch finished, so all notes in a segment
  shared a single, late timestamp. This made rhythm analysis meaningless.
  The loop now keeps an audio clock (`samples_seen`), records
  `segment_start` at onset, and sets
  `event time = segment_start + first note start inside the segment`.
  `TranscriptionBuffer` no longer subtracts a wall-clock start time.
  `analyse_segment(audio, segment_start)` signature changed.
- The basic-pitch model loads once (`_get_basic_pitch_model()`) instead of on
  every segment.
- Duplicate pitches inside one chord group are merged, keeping the loudest.
  Notes are sorted by pitch.
- The note field `confidence` (which was actually velocity/127) is replaced
  by `velocity` (int 0–127).
- Removed the unused `import time`.

**chord_model.py**
- Fixed two crashes that made the BiLSTM backend always fail, with a silent
  fallback to librosa: `self.h` → `self.H` in `run_sequence`, and the
  `hop_lenght` parameter typo in `librosa_chroma_cqt`.
- Verified: **untrained, the BiLSTM output is noise** (all labels wrong on a
  synthetic C–Am–F–G progression, ~1/85 confidence). The header comment and
  the startup warning now say so. No `chord_model_weights.npz` exists;
  `train_on_annotations()` only trains the output layer.

**chord_analysis.py**
- `--backend auto` (the default) uses the BiLSTM only if
  `chord_model_weights.npz` exists, and librosa template matching otherwise.
- librosa backend: windows quieter than `SILENCE_REL_RMS` (5%) of the
  loudest window are labelled `N`. Silence used to be classified as chords.
- librosa backend: each window's `end` is now `start + hop`, so segments
  tile instead of overlapping.

**feedback.py**
- `suggestions.appen` → `append`. This crashed whenever note score < 60.
- Key typos fixed: `assessement` → `assessment`,
  `timing_assessement` → `timing_assessment`. The assessments were never
  shown in the .txt report or the agent prompt.
- Beat grid keys `eight` → `eighth` (consistent with transcription.py).
- `avg_confidence` → `avg_velocity`, read from `velocity` with a fallback to
  legacy `confidence × 127`.
- Rhythm: onsets less than `SIMULTANEOUS_SEC` (0.05 s) apart are skipped.
  Gaps over 2 beats count as hesitations and are excluded from the deviation
  average. With fewer than `MIN_IOIS` (3) intervals, there is no
  `timing_score`, and the assessment says "Not enough consecutive notes to
  judge timing" (it used to report a fake "steady" score).
- Harmony: `N` / `no_chord` / `unknown` segments are excluded from the totals.
- `build_suggestions(..., bpm)`: the metronome advice used a hard-coded
  60 BPM. An empty session now gets a "no notes detected" suggestion instead
  of "Good work!". The encouragement requires a real timing score. Fixed
  missing spaces in messages.
- The `.txt` report shows `N/A` for a missing rhythm score. Fixed typos
  ("Suggesions", a missing space in "In scale").
- `--reference` is still unimplemented; it now prints a notice instead of
  being silently ignored.

**agent.py**
- `--dry-run`: prints the system prompt and opening message, with no API
  call and no Azure configuration needed.
- `_first()` helper reads the new key names and the legacy misspelled ones,
  so old feedback files work.
- The prompt shows `Avg loudness (MIDI velocity)` instead of "Avg confidence".
- `[ERROR]` responses are no longer appended to the history sent to the
  model. An error on the opening turn exits.

**New files:** `requirements.txt` (versions from the old venv) and this
`CLAUDE.md`.

**Not tested for real:** live microphone capture and real basic-pitch
inference (tested with stubs). Azure calls were never tested (by design).

## Known issues / limitations

- Sessions recorded before 2026-10-10 have corrupted `time` values (one
  timestamp per segment, delayed). Their rhythm feedback is unreliable. This
  can't be repaired from the JSON; re-transcribe from the `.wav`
  (roadmap 1).
- Note accuracy only means "in the key". Chromatic notes are counted as
  errors, and there is no reference score or alignment.
- The rhythm metric (deviation from the nearest grid value) is crude: with
  10 grid values, the deviation almost never exceeds ~0.25. So "rhythm too
  irregular" and the metronome suggestion (deviation > 0.4) practically
  never trigger. Documented by the xfail
  `test_rhythm_very_irregular_is_flagged`. The BPM is the one the user set,
  not the tempo they played.
- json_to_score.py inserts no rests, so the score ignores timing gaps.
- Progression detection uses chord roots only, so repeated roots break the
  patterns.
- Real-time segmentation drops audio before the RMS onset and blocks the
  loop during inference.

## Roadmap

0. **Foundations:** ~~venv, git, prototypes folder, main.py, Ollama, tests,
   argparse refactor~~ (all done 2026-10-20). Possible next step: CI
   (GitHub Actions running `pytest -m "not slow"`).
1. **Reliable measurement:** run basic-pitch offline on the full WAV after
   each session; real tempo from beat tracking; validate on MAESTRO.
2. **Expressive features, per phrase:** tempo curve and rubato, dynamics
   (level, range, contour), articulation (duration / inter-onset interval),
   attack, register, brightness (spectral centroid).
3. **Emotion → cue profiles** (Juslin & Laukka; Gabrielsson & Lindström).
   *Happy*: fast, major, staccato, fairly loud, bright, low timing
   variability. *Sad*: slow, legato, soft, dark, more rubato. The learner
   states an intention; the system finds the phrases with the largest gaps
   from that profile. Later, train a classifier on EMOPIA.
4. **A real agent:** tools (phrase features, compare to the target, suggest
   an exercise); prompt focused on expressive coaching rather than error
   correction; Socratic questions; a replay-and-compare loop; a learner
   model and memory across sessions.
5. **Evaluation:** do the measured cues agree with listeners' perception?
   Compare with expert teachers; run a user study.
