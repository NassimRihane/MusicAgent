# main.py
#
# Runs the whole pipeline in one command:
#
#   1. transcription.py        record from the microphone (Ctrl+C to stop),
#                              or --audio FILE; basic-pitch on the full recording
#   2. chord_analysis.py       chords and Roman numerals (music21, or --chords rf)
#   3. json_to_score.py        sheet music (optional, --score)
#   4. feedback.py             correction feedback (notes, rhythm, harmony)
#   5. expressive_feedback.py  emotion heard vs intended (--emotion), with advice
#   6. agent.py                chat with the teacher assistant
#
# Usage:
#   python main.py --emotion sad                                   # record, then coach toward "sad"
#   python main.py --audio "Happy Birthday.mp3" --emotion sad      # analyse a recording
#   python main.py --session sessions/session_20260101_120000.json --emotion happy
#   python main.py --session sessions/session_X.json --emotion calm --dry-run
#   python main.py --scale G --type minor --bpm 80 --emotion tender --lang fr


import argparse
import subprocess
import sys
import time
from pathlib import Path



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Record, analyse and discuss a practice session")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--session", default=None,
        help="Existing session JSON to analyse (skips recording)")
    source.add_argument("--audio", default=None,
        help="Audio file to analyse instead of recording")
    parser.add_argument("--emotion", default=None,
        help="Intended emotion (happy, sad, calm, tense, ...) for the expressive feedback")
    # Recording options (passed to transcription.py)
    parser.add_argument("--scale",  default=None, help="Intended key's tonic (default: detected)")
    parser.add_argument("--type",   default="major", choices=["major", "minor"], help="Intended key's mode")
    parser.add_argument("--bpm",    type=float, default=None, help="Intended tempo (default: estimated)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    # Analysis options
    parser.add_argument("--chords", default="music21", choices=["music21", "rf"],
        help="Chord recogniser: music21 (default, from notes) or rf (Random Forest on audio)")
    parser.add_argument("--no-chords", action="store_true", help="Skip chord analysis")
    parser.add_argument("--score",     action="store_true", help="Also export a MusicXML score")
    # Agent options (passed to agent.py)
    parser.add_argument("--lang",     default="en", choices=["en", "fr"], help="Conversation language")
    parser.add_argument("--provider", default=None, choices=["ollama", "azure"],
        help="LLM provider (default: the one set in agent.py)")
    parser.add_argument("--model",    default=None, help="Ollama model (default: the one set in agent.py)")
    parser.add_argument("--dry-run",  action="store_true",
        help="Print the agent prompt instead of calling the model")
    return parser.parse_args(argv)


ROOT         = Path(__file__).parent
SESSIONS_DIR = ROOT / "sessions"



# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────

def run_step(title: str, script: str, *script_args) -> bool:
    """
    Run one pipeline script with the current Python interpreter.
    Ctrl+C reaches the child (e.g. to stop recording); the pipeline itself
    keeps waiting so the child can save its files and exit cleanly.
    Returns True if the script succeeded.
    """
    print(f"\n{'#' * 60}\n#  {title}\n{'#' * 60}", flush=True)
    cmd  = [sys.executable, str(ROOT / script), *[str(a) for a in script_args]]
    proc = subprocess.Popen(cmd)
    while True:
        try:
            return proc.wait() == 0
        except KeyboardInterrupt:
            continue


def newest_session(since: float):
    """Most recent session JSON written after `since` (epoch seconds), or None."""
    candidates = [p for p in SESSIONS_DIR.glob("session_*.json")
                  if not p.stem.endswith("_chords") and p.stat().st_mtime >= since]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def emotion_model_ready() -> bool:
    import emotion_model
    return emotion_model.MODEL_PATH.exists()



# ─────────────────────────────────────────────
# 2. MAIN
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)

    # Step 1 — recording, audio file, or an existing session
    if args.session:
        session = Path(args.session)
        if not session.exists():
            print(f"ERROR: file not found — {session}")
            sys.exit(1)
    else:
        rec_args = ["--type", args.type]
        if args.scale:
            rec_args += ["--scale", args.scale]
        if args.bpm:
            rec_args += ["--bpm", args.bpm]
        if args.audio:
            rec_args += ["--audio", args.audio]
        if args.device is not None:
            rec_args += ["--device", args.device]
        title = "1/5  Transcription" if args.audio else "1/5  Recording  (press Ctrl+C to stop)"
        started = time.time()
        run_step(title, "transcription.py", *rec_args)
        session = newest_session(started)
        if session is None:
            print("ERROR: no session file was saved — nothing to analyse.")
            sys.exit(1)

    # Step 2 — chords (chord_analysis.py falls back to music21 if there is no WAV)
    if args.no_chords:
        print("\n  Chord analysis skipped (--no-chords).")
    elif not run_step("2/5  Chord analysis", "chord_analysis.py", session, "--backend", args.chords):
        print("  Chord analysis failed — continuing without harmony feedback.")

    # Optional — sheet music
    if args.score:
        run_step("     Score export", "json_to_score.py", session)

    # Step 3 — correction feedback
    if not run_step("3/5  Correction feedback", "feedback.py", session):
        print("ERROR: feedback step failed.")
        sys.exit(1)
    feedback = session.parent / f"feedback_{session.stem}.json"

    # Step 4 — expressive feedback (needs an intention and the trained emotion model)
    if not args.emotion:
        print("\n  No --emotion given: the coach will only use the correction feedback.")
    elif not emotion_model_ready():
        print("\n  Emotion model not trained yet. Run once:  python emotion_model.py --train")
        print("  Continuing with the correction feedback only.")
    elif not run_step("4/5  Expressive feedback", "expressive_feedback.py",
                      session, "--emotion", args.emotion):
        print("  Expressive feedback failed — continuing with the correction feedback only.")

    # Step 5 — conversation
    agent_args = [feedback, "--lang", args.lang]
    if args.provider:
        agent_args += ["--provider", args.provider]
    if args.model:
        agent_args += ["--model", args.model]
    if args.dry_run:
        agent_args.append("--dry-run")
    run_step("5/5  Teacher assistant", "agent.py", *agent_args)


if __name__ == "__main__":
    main()
