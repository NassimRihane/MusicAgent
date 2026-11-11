# CLAUDE.md

Context for Claude Code sessions on this project. Read it first; update the
**Change log** and **Known issues** sections whenever you change the code.

## Project goal

A research prototype (Keio) for an AI assistant teacher for piano learners.
The goal is to **guide the learner toward playing expressively**, not only
to correct mistakes: the learner states an intention (e.g. *sad*), the
system estimates which emotion the performance conveys, and coaches the
learner on the playing choices that would close the gap.

## Architecture rule (from the user's supervisor, 2026-11-07)

**Only the expressive feedback (layer 4) may be original. Everything else
must use widely used, published tools or textbook methods ("only one degree
of freedom").** The supervisor was confused by a custom-looking ML model, so:

- **Never add custom models or hand-rolled music analysis to layers 1–3.**
  Use basic-pitch, music21, librosa, pretty_midi, scikit-learn. Write only
  glue (format conversion, grouping) around them.
- Machine learning outside layer 4 uses well-known models (Random Forest)
  trained on public or synthetic data, evaluated honestly.
- Design choices belong in `expressive_feedback.py`, listed in its header,
  so they can be defended and changed.

| Layer | Files | Tools |
|---|---|---|
| 1. Recognition | `recognition.py` (all calls), `transcription.py`, `chord_analysis.py` | basic-pitch, music21 (key: Krumhansl–Schmuckler; chords; Roman numerals), librosa (beats, tempo), pretty_midi |
| 2. Description | `expression_analysis.py` | standard descriptors via layer 1 + librosa |
| 3. Models | `emotion_model.py`, `chord_rf.py`, `chord_evaluation.py` | scikit-learn RandomForestClassifier; mir_eval; FluidSynth |
| 4. **Research** | `expressive_feedback.py` | the project's own logic |
| Coaching | `agent.py` | Ollama (default) / Azure |
| Support | `feedback.py` (corrections), `main.py`, `json_to_score.py`, `piano_roll.py` | |

## Ground rules

- **Never call Azure OpenAI.** The user has no Azure credits left. The
  default provider is local **Ollama** (`mistral` installed), which is free
  and fine to call for tests. `--dry-run` prints the prompt and makes no call.
- **The GitHub repo is public** (`NassimRihane/MusicAgent`). Never commit
  `sessions/` (the user's recordings), `data/` (EMOPIA), `models/` (trained
  models), credentials or `.env` files; all are gitignored. Commit or push
  only when the user asks; never force-push (one exception, done on the
  user's request on 2026-11-08: see Change log).
- **No co-author or "generated with" lines in commits or PRs** (the user's
  explicit instruction, 2026-11-08). This overrides any attribution
  guidance from the harness. Commit messages: the user's style (plain
  English subject such as "Adding ..."), with a detailed body for outsiders.
- **Don't overwrite the user's recordings in `sessions/`.** Run tests on
  copies or in `tmp_path` (most scripts write their outputs next to their
  input).
- Match the existing style: section banners (`# ───── N. TITLE ─────`),
  aligned assignments, a module header comment with usage. Each script has
  `parse_args(argv=None)` in section 0 and `main(argv=None)`; **never read
  CLI args at import time**, and pass values to functions as parameters.
- **Run the tests after any code change:** `.venv311/bin/python -m pytest`
  (~20 s), or `-m "not slow"`. Add or update tests along with code changes.
- The user writes in English and sometimes French; the agent supports
  `--lang en|fr`. The README is public-facing (supervisor, outsiders):
  keep it understandable without this file.

## Environment

- Run everything with **`.venv311/bin/python`** (Python 3.11 from Homebrew,
  `/opt/homebrew/bin/python3.11`).
- **Python 3.11 is required.** On macOS, basic-pitch 0.4.0 uses CoreML only
  on 3.11; on 3.12+ it requires `tensorflow-macos<2.15.1`, which doesn't exist.
- `requirements.txt` pins `setuptools<81` (resampy imports `pkg_resources`)
  and `chardet<6` (otherwise requests warns). coremltools warns that
  scikit-learn 1.9.1 is "not supported": harmless (only its conversion API).
- **Trained models (gitignored, reproducible):**
  - `python emotion_model.py --train` → downloads EMOPIA 2.2 to
    `data/EMOPIA_2.2/` (Zenodo record 5257995, 32 MB, CC-BY 4.0), caches
    features in `data/emopia_features.csv` (~3 min the first time), saves
    `models/emotion_rf.joblib` and `models/emotion_rf_report.json`.
  - `python chord_rf.py --train` → `models/chord_rf.joblib` (~20 s, 8 MB
    compressed; also trained automatically on first use).
  - `python chord_evaluation.py` (~30 s) → `models/chord_eval_report.{json,md}`.
    `--recording take.wav` scores a real take of `protocols/chord_recording.md`
    (28 chords: 7 qualities × 4 roots, struck once, held ~2 s, ~1 s silence;
    source of truth `protocols/chord_recording.json`) →
    `models/chord_eval_recording_<take>.{json,md}`. `--render-protocol out.wav`
    renders the expected take. Segmentation: `librosa.effects.split`
    (top_db 35), fallback to the N strongest onsets.
    Needs FluidSynth (`brew install fluid-synth`, installed),
    `pyfluidsynth` and the GeneralUser GS soundfont in
    `data/soundfonts/GeneralUser-GS.sf2` (32 MB, free licence, downloaded;
    its licence file is next to it).
- Ollama runs locally at `http://localhost:11434`; installed model: `mistral`.
- Git: `main` tracks `origin` = https://github.com/NassimRihane/MusicAgent
  (public). `core.fileMode=false`, because every file on disk is mode 755.
  No `gh` CLI. macOS has no `timeout` command: use
  `perl -e 'alarm N; exec @ARGV' cmd`.

## Tests

`tests/`, one file per module; config in `pytest.ini`; `pytest` is in
`requirements-dev.txt`. 173 tests, ~30 s.

- `conftest.py` blocks `requests.post` in every test (no Ollama or Azure
  calls; agent tests install fakes). It provides synthetic audio (`tones`,
  `silence`), session builders (`event`, `make_session`, `write_json`) and
  synthetic emotion-feature rows (`synthetic_rows`, `PROFILE`). Tests write
  only to `tmp_path` and never need `data/` or `models/`: models are trained
  on small synthetic data inside the tests (`test_emotion_model.py` even
  writes a miniature EMOPIA with real MIDI files).
- An autouse fixture sets `emotion_model.N_JOBS = chord_rf.N_JOBS = 1`:
  Random Forests with `n_jobs=-1` spawned worker processes that made test
  runs erratically slow (one run hung > 5 min). Emotion tests also use
  `N_TREES = 50`.
- `slow` marker: real basic-pitch on synthesised notes, and the full
  `main.py --session … --dry-run` pipeline as subprocesses.
- Mutation-checked on 2026-11-07: reintroducing bugs (event time from the
  lowest note, minor keys without raised 6th/7th, no chord-window extension,
  flipped advice direction, inert what-if, cross-validation not grouped by
  song) makes tests fail.

## Pipeline

`main.py` runs every step in order; each step also runs on its own:

```
transcription.py        mic (live preview) or --audio FILE → session_X.wav
                        → basic-pitch on the FULL recording, key (music21),
                          tempo (librosa) → session_X.json
        ├─ chord_analysis.py   → session_X_chords.json/.txt
        │                        --backend music21 (default) | rf (needs WAV) | both
        ├─ json_to_score.py    → .musicxml / .pdf / .mid (optional, --score)
        ▼
feedback.py             correction feedback (notes in key, tempo stability,
                        harmony) → feedback_session_X.json/.txt
expressive_feedback.py  --emotion E → expressive_session_X.json
        ▼
agent.py                feedback_*.json (+ expressive_*.json, auto-found)
                        → chat → conversation_feedback_session_X.json
```

`main.py` options: `--audio FILE` / `--session JSON` / (record),
`--emotion`, `--scale/--type/--bpm` (declared key/tempo; default detected),
`--chords music21|rf` (default music21), `--no-chords`, `--score`, `--lang`, `--provider`,
`--model`, `--dry-run`. The expressive step is skipped with a message if
`models/emotion_rf.joblib` is missing.

Other tools: `expression_analysis.py A --compare B --same-piece` (happy→sad
cue directions from Juslin & Laukka 2003, with `MIN_CHANGE` thresholds);
`emotion_model.py --predict FILE`; `piano_roll.py` (matplotlib piano rolls of
audio / session / expression JSON).

Model selection for the chat: the `PROVIDER` / `MODEL` constants at the top
of `agent.py`, or `--provider` / `--model`. Both providers use the OpenAI
chat format (Ollama via `/v1/chat/completions`).

Legacy prototypes in `prototypes/` (not part of the pipeline):
`note_detection.py`, `chord_detection.py`, `sim_proba.py` (a Bayesian
simulation of learner motivation that could later feed a learner model).

### Key definitions

- **Note** (everywhere): `{"start": s, "end": s, "midi": int, "velocity": 0–127}`.
- **Key/scale**: `recognition.scale_pitch_classes` uses music21; minor keys
  include the harmonic and melodic minor (raised 6th/7th), so G# in A minor
  is in key.
- **Chords**: `recognition.chords_from_notes` names the notes sounding in
  each beat with music21 (weights = overlap × velocity; ≤ 4 pitch classes
  with ≥ 20 % of the max weight). A window with < 3 pitch classes is
  extended up to 3 beats. Canonical labels (`"C"`, `"A:min"`, `"G:dom7"`)
  are shared with `chord_rf.VOCAB`.
- **Tempo stability** (`tempo_cv` in `track_beats`): CV of librosa's local
  tempo, ignoring the first/last 4 s and folding octave errors to the
  median (MIREX practice). Steady 0; 20 ms jitter ~0.02; ritardando
  120→67 BPM ~0.11. `feedback.py`: < 0.03 steady, < 0.08 mostly steady,
  0.15 scores 0.
- **Emotion features** (`expression_analysis.NOTE_FEATURES`, identical on
  EMOPIA MIDI and on our transcriptions): tempo_bpm, onset_rate,
  timing_variability, articulation, velocity_mean, velocity_std,
  mean_pitch, pitch_range (p5–p95), polyphony, major_minor. EMOPIA notes
  are read with the sustain pedal applied (`load_midi_notes(sustain=True)`).
- **Quadrants**: Q1 happy/excited, Q2 tense/angry, Q3 sad, Q4 calm/relaxed;
  `emotion_model.EMOTION_TO_QUADRANT` maps learner words.

### Session JSON schema (transcription.py output)

```json
{"meta": {"scale": "C major", "scale_source": "declared | detected (music21)",
          "key_detected": "C major", "key_correlation": 0.79,
          "bpm": 112.3, "bpm_source": "declared | estimated (librosa)",
          "tempo_estimated": 112.3, "date": "...", "audio": "session_X.wav",
          "source": "microphone | <file name>", "transcription": "basic-pitch (full recording)"},
 "events": [{"type": "note|chord", "time": 0.5, "duration": 0.4,
             "rhythm": "eighth", "in_scale": true,
             "notes": [{"midi": 60, "name": "C4", "pitch_class": 0,
                        "in_scale": true, "velocity": 90}]}],
 "notes": [{"start": 0.5, "end": 0.9, "midi": 60, "velocity": 90}]}
```

- `notes` is the source of truth; `events` (grouped attacks) is kept for
  `json_to_score.py` and older readers. `recognition.session_notes()` reads
  either (sessions before 2026-11-07 have no `notes`; before 2026-10-10 they
  store `confidence` = velocity/127 and have broken `time` values).

## Change log

### 2026-11-12 (b): Mid-term deck draft

- `presentation/midterm_draft.pptx` (gitignored): 10 slides + 3 backup,
  English, speaker notes on every slide, native charts and tables (for
  editing after import into Canva: Create a design → Import file). No Canva
  connector in Claude Code. Built by `presentation/build_deck.js`
  (pptxgenjs; rebuild: `NODE_PATH=<dir with pptxgenjs>/node_modules node
  presentation/build_deck.js <project root> <out.pptx>`; numbers are
  hard-coded from the reports of 2026-11-12). Visual QA done (LibreOffice
  render). Installed for this: Homebrew `node`, `poppler`, LibreOffice
  cask; pptxgenjs 3.12.0 in the scratchpad.
- Slide order: title · apps vs teacher + research goal · 4 layers ·
  scope/exclusions · emotion model (quadrants, RF vs baselines) · advice
  design (Clair de lune normal, sad 25 → 42 %) · demo two takes (5/8 cues,
  model 25 → 27 %) · finding: velocity ignores level · chords as an example
  of justified choice · next steps (dark). Backup: confusion, importance,
  chords by type.


### 2026-11-12: Second sad take; basic-pitch velocity is level-invariant

- `clair_de_lune_sad2.wav` (played again by the user): slower (performance
  41.85 → 46.93 s, 89 → 80 BPM), more rubato (timing_variability 0.81 →
  0.88, feedback flags "tempo varies"), recording 12.6 dB quieter, less
  dynamic range in the audio (24.9 → 9.3 dB), lower register; literature
  check (expression_analysis, Juslin directions): 5 agree (slower,
  quieter recording, lower register, more rubato, less dynamic range),
  2 no change (mode, articulation), 1 opposite (brightness). But the RF barely moves (sad 25 → 27 %, heard
  Q1 28 % / Q3 27 % / Q4 28 %) because velocity_mean is unchanged
  (77.6 → 77.5); the advice still says "play softer".
- **Finding: basic-pitch's velocity ignores the recording level.** Same
  take scaled by 0/−12/−24 dB → mean velocity 77.8 / 79.9 / 81.2. Velocity
  measures dynamics within a take, not overall softness, so the model's top
  cue can't hear a uniformly softer performance, and "play softer" advice is
  unreliable. docs/decisions.md §4 corrected (the earlier "loudness means
  touch" statement was wrong). Fix planned after the mid-term: controlled
  recording setup + relate audio level to the model.
- Demo files: `expression_sad2.json` (Normal vs Sad version 2),
  `expressive_clair_de_lune_sad2.json`; `clair_de_lune_sad.wav` (first sad
  take) differed from the normal one mainly by recording level.


### 2026-11-11: Clair de lune demo (the user's own playing)

- `sessions/demo_clair_de_lune/`: `clair_de_lune.wav` (normal) and
  `clair_de_lune_sad.wav` (played again by the user, trying to sound sad),
  analysed with clear names (`clair_de_lune.json`, `feedback_…`,
  `expressive_…` with `--emotion sad`, `expression.json` comparing the two
  with labels Normal / Sad version). `first_run/` holds the user's own
  earlier run (`session_20261111_184056*`). It's a simplified arrangement
  in C major (detected), not Debussy's D♭ original.
- Results: normal → heard Q4 calm (32 %), sad 25 %; advice: softer touch,
  more even dynamics, steadier pulse; what-if 25 % → 42 %. The sad take is
  heard the same (calm 30 %, sad 24 %, same advice): timing almost
  identical (performance 41.85 vs 41.87 s), velocity 77.6 vs 80.4, but
  recording level −32.5 vs −43.2 dB. So the user changed mostly the level,
  not the touch or tempo.
- Figures for this demo: `figures/clair_de_lune/` (advice_example,
  piano_roll_clair_de_lune).
- Presentation: 10-minute mid-term talk in English; slides due 2026-11-12.
  No Canva connector is available in Claude Code here; the plan is a .pptx
  (or a claude.ai Slides deck downloaded as .pptx) imported into Canva.


### 2026-11-10 (b): music21 default again; figures; decisions document

- **Chords: music21 is the default again** (`chord_analysis.py`, `main.py`),
  decided with the user from the real recording: RF never outputs sevenths
  or dims on real audio (systematic bias), music21's errors aren't
  systematic, and chords don't feed the expressive advice. RF stays as
  `--chords rf` / `--backend rf`; `--backend both` saves music21's result.
- **`make_figures.py` (new):** regenerates figures from the saved reports
  into `figures/` (gitignored: reproducible, some derive from third-party
  audio), PNG + PDF + `captions.md` with the current numbers:
  emotion_model_comparison, emotion_confusion, emotion_feature_importance,
  chord_eval_comparison, chord_eval_by_quality, advice_example,
  piano_roll_demo. Colours: music21 / highlighted = #2a78d6, RF = #eb6834,
  others neutral grey. All checked visually.
- **`docs/decisions.md` (new):** the research goal (agreed with the user:
  "design and evaluate how a system should give expressive advice…"),
  scope and exclusions with justification (background noise, recognition
  accuracy, real time, other instruments, LLM comparison), deferred work,
  and one entry per design choice (decision, why, evidence, alternatives,
  limits). **Keep it updated whenever a choice changes**: the supervisor
  asked that every choice and exclusion be justified.
- Tests: `test_make_figures.py`; 173 tests.


### 2026-11-10: First real recording; supervisor feedback; new schedule

- **Real chord take** (`sessions/session_20261109_163000.wav`, the 28-chord
  protocol, report `models/chord_eval_recording_session_20261109_163000.md`):
  RF root 100 %, majmin 100 %, sevenths 45 % (it never outputs a 7th chord
  and turns every dim into min/maj); basic-pitch + music21 root 93 %,
  majmin 100 %, sevenths 95 % (fails only on the 2 augmented chords, which
  are ambiguous by nature, and chord 1 heard as C7). Opposite of the
  rendered-progression test (RF far better in context).
- **Supervisor feedback (to apply everywhere):** compare models in the
  performance report; set a final goal and **justify every choice and every
  exclusion** (e.g. if the research is the design of the advice, say why
  background-noise robustness is out of scope). Measuring "ability to give
  advice" as the performance criterion for choosing models can come later.
- **Schedule:** mid-term presentation 2026-11-20; slides due 2026-11-12 for
  a training session. The user does not want Claude to make the slides
  now, only the work behind them.


### 2026-11-08 (b): Real-recording chord test; co-author lines removed from history

- `protocols/chord_recording.{json,md}`: 28 chords for the user to record
  (both hands, root position; flats for flat-key chords). E augmented was
  replaced by D augmented: augmented triads are symmetric (C/E/G# aug share
  their notes), so their root can't be heard; noted in the sheet.
- `chord_evaluation.py --recording / --render-protocol` (see Environment).
  Self-check on the rendered protocol (isolated held chords): RF root 96 %,
  majmin 100 %, sevenths 64 %; music21 93 / 100 / 82 %. The RF tends to
  drop the 7th on isolated chords; in context (progressions) it is far
  better. No real take recorded yet.
- **Git history rewritten on the user's request:** the "Co-Authored-By:
  Claude" trailer was removed from every commit message
  (`git filter-branch --msg-filter`), then force-pushed. All commit hashes
  changed; a local backup branch `backup/before-author-rewrite` holds the
  old history.


### 2026-11-08: Evaluations for the paper; Random Forest becomes the default chord recogniser

The user checked the supervisor's requirements (number of emotions, standard
methods, accuracy numbers, famous model, dev finished within 2026).

- **Emotions:** 4 quadrants of Russell's valence–arousal model (EMOPIA);
  they match Juslin's basic performance emotions (happiness, anger,
  sadness, tenderness).
- **emotion_model.py:** the report now compares the Random Forest with
  standard classifiers on the same song-grouped folds (`baseline_models`,
  `compare_models`, `cross_validate`, `scores`), with mean ± std over
  folds. Results: majority 28.8 %, logistic regression 65.9 %, k-NN 64.7 %,
  SVM 67.2 %, RF 67.9 % ± 3.1. Training also writes
  `models/emotion_rf_report.md` (tables) and `models/emotion_confusion.png`.
- **chord_evaluation.py (new):** the chord RF had no credible accuracy (only
  synthetic held-out). Test set: progressions in random keys/tempos, 3
  textures (block, arpeggio, melody), rendered with FluidSynth +
  GeneralUser GS (different sounds from the training synth), beats from
  librosa as in the pipeline, MIREX metrics with mir_eval. 36 pieces:
  **RF root 91.9 %, majmin 82.1 %, sevenths 70.5 %; basic-pitch + music21
  52.1 / 52.2 / 36.2 %.** Diagnosis of music21's weakness: basic-pitch ends
  decaying piano notes early, so music21 names incomplete chords ("Cpedal",
  intervals); 17 % of its segments fall outside the triad/7th vocabulary.
- **The chord RF is now the default** (`chord_analysis.py --backend rf`,
  `main.py --chords rf`), falling back to music21 when a session has no WAV.
  music21 still names RF labels and gives Roman numerals.
- chord_rf: `min_samples_leaf=2` and compressed joblib (194 MB → 8 MB);
  held-out synthetic 94.3 %.
- requirements: `pyfluidsynth==1.4.0`.
- Tests: `test_chord_evaluation.py` (render replaced by sine tones, no
  soundfont needed); emotion tests redirect REPORT_MD and FIGURE_PATH to
  tmp (they would otherwise overwrite the real report).


### 2026-11-07: Restructure into standard layers + Random Forests (supervisor's request)

The supervisor asked that everything except the expressive feedback rely on
widely used tools, and suggested well-known models such as Random Forest.
Decisions taken with the user: Random Forest for **both** emotion and chord
recognition; chords named by **music21 from the notes** by default (the RF
is the audio alternative); correction feedback **kept, on standard tools**.

- **recognition.py (new, layer 1):** all calls to basic-pitch (model loaded
  once, debug prints silenced), pretty_midi (with sustain pedal), music21
  (key, scale, chord symbols, Roman numerals) and librosa (beats, tempo
  stability). Replaces the hand-written scale tables, Krumhansl–Schmuckler
  code and BEAT_GRID duplicated across files.
- **chord_model.py deleted** (custom numpy BiLSTM, never trained; in git
  history at 49f90ea). **chord_rf.py (new):** Random Forest (300 trees) on
  beat-synchronous CQT chroma, trained on 3,400 synthetic piano chords
  (85 classes): 94.6 % held-out on synthetic data, 59–70 % agreement with
  music21 on the scale degree on the Happy Birthday clips (confuses major
  and minor).
- **chord_analysis.py rewritten:** input is a session JSON; backends
  `music21` (default), `rf`, `both` (reports agreement). The librosa
  template backend is gone.
- **emotion_model.py (new):** Random Forest (500 trees, balanced classes) on
  EMOPIA. GroupKFold by song: **67.9 %** 4-quadrant accuracy (chance
  28.9 %), arousal 88.7 %, valence 75.4 %. Permutation importance:
  velocity_mean 0.24, major_minor 0.15, all others < 0.03; tempo_bpm and
  articulation ≈ 0. On the Happy Birthday clips: happy → Q1 (39 %),
  sad → Q3 (52 %).
- **expressive_feedback.py (new, layer 4):** see its header. Gap = (value −
  target median) / target IQR on actionable cues (tempo, loudness, dynamic
  contrast, articulation, timing freedom); priority = |gap| × permutation
  importance; top 3 with |gap| ≥ 0.5 become advice; what-if re-prediction.
  Piece-fixed cues (mode, register, range, polyphony, density) are reported,
  never advised.
- **expression_analysis.py:** now layer 2 only; `note_features()` shared
  with the emotion model; key via music21 (the hand-rolled KS is gone).
- **transcription.py rewritten:** the session is built by basic-pitch on the
  full recording (`build_session`); the segment-by-segment transcription is
  now only a live preview. `--audio FILE` analyses an existing recording.
  `--scale` and `--bpm` are optional (default: detected / estimated).
  Sessions now store `notes`. TranscriptionBuffer removed.
- **feedback.py rewritten:** notes in key via music21 (with minor
  variants); rhythm = librosa tempo + tempo stability + hesitations (the
  beat-grid deviation was tried and **dropped**: arpeggios and dotted
  rhythms made the real happy clip score worse than random notes);
  harmony from Roman numerals with major and minor progression patterns.
  The metronome advice now says "if this wasn't intentional rubato".
- **agent.py:** new expression-first prompts (EN/FR): intention first,
  numbers turned into musical language, never ask to remove what the
  expressive analysis asks for (e.g. rubato), never change piece-fixed
  cues. `expressive_<session>.json` is found automatically (`--expressive`
  to override); the opening states the intention; the log stores it.
  The LLM gets gaps in words, not raw values (`describe_gap`: slightly /
  clearly / much + louder, slower, more detached…): with raw numbers,
  mistral recited "your loudness is 66.2, typical 54.56" to the learner.
- **main.py:** 5 steps; `--audio`, `--emotion`, `--chords`; no default key
  or tempo.
- **piano_roll.py:** uses recognition.py (session `notes`, music21 key).
- Fixed during testing: `notes_to_events` took a chord's time from its
  lowest note instead of its earliest onset.
- Tests rewritten: 150 tests (new `test_recognition.py`,
  `test_emotion_model.py`, `test_expressive_feedback.py`); the old xfail
  is gone with the old rhythm metric.
- requirements: `scikit-learn==1.9.1`, `pretty_midi==0.2.11.post0`.
  `.gitignore`: `data/`, `models/`.
- Example sessions made from the demo clips (gitignored):
  `sessions/session_20261107_164559*` (Happy Birthday, analysed with
  `--emotion sad`) and `sessions/session_20261107_164234*` (Unhappy
  Birthday, `--emotion happy`).

### 2026-10-21 (b): piano_roll.py

- New matplotlib script (see Pipeline) so the "What the system heard" view
  can be regenerated for any extract without a new artifact.
  `tests/test_piano_roll.py` (suite: 101 passed + 1 xfail).
  `matplotlib==3.11.2` pinned in requirements (it was already installed
  via music21).
- Demo image: `sessions/demo_happy_birthday/piano_roll.png`.
- Old sessions (before 2026-10-10) visibly show the timestamp bug in the
  roll: notes are stacked in columns every ~4 s.

### 2026-10-21: Expressive cue analysis + supervisor demo

- **expression_analysis.py (new):** first version of roadmap step 2 (see
  Pipeline). Design choices made after checking real data:
  - Mode comes from a duration × velocity pitch-class histogram of the
    *transcribed notes*. Audio chroma called the sad clip "A major" (0.83
    vs 0.80 for A minor), because overtones of A and E hid the third.
  - For two takes of the same piece, tempo = time from first to last
    onset. Beat tracking reported only −4 % against a real +31 %.
  - Changes smaller than `MIN_CHANGE` count as "no clear change", so that
    tiny differences are not reported as matches.
- Tests: `tests/test_expression_analysis.py` (suite now 91 passed + 1 xfail).
- **Demo for the user's supervisor** (2026-10-22 meeting): two YouTube clips
  (*When You Turn HAPPY SONGS into SUPER SAD*), stored outside the repo at
  `~/Documents/Audacity4/{Happy,Unhappy} Birthday.mp3`. Results in
  `sessions/demo_happy_birthday/` (gitignored): `expression.json`, and
  `coach_answer.json` (unedited prototype answer from local mistral).
  Result: 5/8 cues agree; register and brightness show no clear change;
  dynamic contrast is the opposite. Published as a private artifact,
  "Happy vs Unhappy Birthday":
  https://claude.ai/artifact/9V5YkJWkBLDUJUSXTYWvzQ (the page's source
  was a scratchpad file; the data is in the JSON above). The audio is
  third-party, so it is not embedded in the page or committed.

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

- **No score, so no rhythm correctness.** Tempo stability can't tell
  intentional rubato from an unsteady pulse. `tempo_cv` is also noisy on
  short sequences with random jitter (same jitter, CV 0.01–0.10 across
  seeds); it reliably measures real tempo changes. Score-informed rhythm
  (alignment to a reference with DTW) is on the roadmap.
- **EMOPIA domain gap:** pop piano, transcribed by a different system
  (GiantMIDI / Kong et al.) than basic-pitch, with pedal-extended notes.
  Typical articulation for Q3 in EMOPIA is ~3.6 vs ~1.3 in basic-pitch
  transcriptions, so articulation advice has the right direction but an
  exaggerated size. Velocity calibration may differ too, and velocity
  dominates the model (it may partly reflect recording loudness).
- tempo_bpm and articulation have ~0 importance in the emotion model, so
  they rarely rank high in the advice even when the gap is large.
- The chord evaluation uses rendered piano (FluidSynth), not real
  recordings; a small real set (the user playing known chords) is planned.
  The RF's errors are mostly major/minor and sevenths. The music21 backend
  (notes) is much weaker (see 2026-11-08) and its names include melody
  notes ("C/GaddD").
- Note accuracy only means "in the key"; chromatic notes count as errors.
- json_to_score.py inserts no rests, so the score ignores timing gaps.
- Live preview segmentation drops audio before the RMS onset and blocks the
  loop during inference (the saved session is unaffected: it uses the full
  recording).
- **The recording path (microphone) is untested** since the 2026-11-07
  rewrite of transcription.py; `--audio` is tested end to end.

## Roadmap

0. **Foundations:** done (venv, git, tests, main.py, Ollama, standard
   layers). Possible: CI (GitHub Actions running `pytest -m "not slow"`).
1. **Score-informed analysis:** align the performance to a reference
   (MusicXML/MIDI) with DTW (librosa.sequence.dtw) for real note and rhythm
   correctness, and phrase boundaries.
2. **Phrase-level expressive feedback:** run layer 4 per phrase/window to
   say *where* the expression falls short.
3. **Reduce the EMOPIA domain gap:** normalise velocities per performance,
   or transcribe EMOPIA audio with basic-pitch; validate on a few recorded
   happy/sad takes by the user.
4. **Agent:** tools (re-analyse a phrase, compare two takes), a
   replay-and-compare loop, a learner model and memory across sessions.
5. **Evaluation:** do the model's judgements match listeners'? Compare the
   advice with expert teachers; user study.
