# main.py
#
# Runs the whole pipeline in one command:
#
#   1. transcription.py   record from the microphone (Ctrl+C to stop)
#   2. chord_analysis.py  chords from the session WAV
#   3. json_to_score.py   sheet music (optional, --score)
#   4. feedback.py        pedagogical feedback
#   5. agent.py           chat with the teacher assistant
#
# Usage:
#   python main.py --scale C --type major --bpm 60
#   python main.py --session sessions/session_20260101_120000.json   (skip recording)
#   python main.py --session sessions/session_20260101_120000.json --dry-run
#   python main.py --scale G --type minor --bpm 80 --lang fr --model llama3.1


import argparse
import subprocess
import sys
import time
from pathlib import Path



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Record, analyse and discuss a practice session")
parser.add_argument("--session", default=None,
    help="Existing session JSON to analyse (skips recording)")
# Recording options (passed to transcription.py)
parser.add_argument("--scale",  default="C", help="Root note of the scale (e.g. C, G, F#)")
parser.add_argument("--type",   default="major", choices=["major", "minor"], help="Scale type")
parser.add_argument("--bpm",    type=float, default=60.0, help="Tempo in BPM")
parser.add_argument("--device", type=int, default=None, help="Audio input device index")
# Analysis options
parser.add_argument("--no-chords", action="store_true", help="Skip chord analysis")
parser.add_argument("--score",     action="store_true", help="Also export a MusicXML score")
# Agent options (passed to agent.py)
parser.add_argument("--lang",     default="en", choices=["en", "fr"], help="Conversation language")
parser.add_argument("--provider", default=None, choices=["ollama", "azure"],
    help="LLM provider (default: the one set in agent.py)")
parser.add_argument("--model",    default=None, help="Ollama model (default: the one set in agent.py)")
parser.add_argument("--dry-run",  action="store_true",
    help="Print the agent prompt instead of calling the model")
args = parser.parse_args()


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



# ─────────────────────────────────────────────
# 2. MAIN
# ─────────────────────────────────────────────

def main():
    # Step 1 — recording (or an existing session)
    if args.session:
        session = Path(args.session)
        if not session.exists():
            print(f"ERROR: file not found — {session}")
            sys.exit(1)
    else:
        rec_args = ["--scale", args.scale, "--type", args.type, "--bpm", args.bpm]
        if args.device is not None:
            rec_args += ["--device", args.device]
        started = time.time()
        run_step("1/4  Recording  (press Ctrl+C to stop)", "transcription.py", *rec_args)
        session = newest_session(started)
        if session is None:
            print("ERROR: no session file was saved — nothing to analyse.")
            sys.exit(1)

    # Step 2 — chords (optional: needs the WAV next to the JSON)
    if args.no_chords:
        print("\n  Chord analysis skipped (--no-chords).")
    elif not session.with_suffix(".wav").exists():
        print(f"\n  No WAV found for {session.name} — chord analysis skipped.")
    elif not run_step("2/4  Chord analysis", "chord_analysis.py", session):
        print("  Chord analysis failed — continuing without harmony feedback.")

    # Optional — sheet music
    if args.score:
        run_step("     Score export", "json_to_score.py", session)

    # Step 3 — feedback
    if not run_step("3/4  Feedback", "feedback.py", session):
        print("ERROR: feedback step failed.")
        sys.exit(1)
    feedback = session.parent / f"feedback_{session.stem}.json"

    # Step 4 — conversation
    agent_args = [feedback, "--lang", args.lang]
    if args.provider:
        agent_args += ["--provider", args.provider]
    if args.model:
        agent_args += ["--model", args.model]
    if args.dry_run:
        agent_args.append("--dry-run")
    run_step("4/4  Teacher assistant", "agent.py", *agent_args)


if __name__ == "__main__":
    main()
