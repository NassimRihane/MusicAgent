# Shared fixtures for the test suite.
#
# Guarantees:
#   - no network: requests.post is blocked in every test (tests that exercise
#     the agent replace it with a fake response)
#   - no microphone and no access to sessions/: audio is synthesised and every
#     file is written to pytest's tmp_path

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 22050


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def blocked(*a, **kw):
        raise RuntimeError("Network access is disabled in tests")
    monkeypatch.setattr(requests, "post", blocked)


# ─────────────────────────────────────────────
# Audio synthesis
# ─────────────────────────────────────────────

def midi_to_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def tones(midis, dur: float, sr: int = SR, amp: float = 0.1) -> np.ndarray:
    """Sum of pure sine tones (a chord) with a short fade in/out."""
    t = np.arange(int(sr * dur)) / sr
    y = sum(np.sin(2 * np.pi * midi_to_hz(m) * t) for m in midis) * amp
    fade = min(len(t) // 10, int(0.01 * sr))
    env = np.ones_like(t)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (y * env).astype(np.float32)


def silence(dur: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * dur), dtype=np.float32)


# ─────────────────────────────────────────────
# Session data builders
# ─────────────────────────────────────────────

def note(midi: int, velocity: int = 80, scale_pc=frozenset({0, 2, 4, 5, 7, 9, 11})) -> dict:
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    pc = midi % 12
    return {"midi": midi, "name": f"{names[pc]}{midi // 12 - 1}",
            "pitch_class": pc, "in_scale": pc in scale_pc, "velocity": velocity}


def event(t: float, midis, rhythm: str = "quarter", duration: float = 0.5) -> dict:
    notes = [note(m) for m in midis]
    return {"type": "chord" if len(notes) > 1 else "note", "notes": notes,
            "duration": duration, "rhythm": rhythm,
            "in_scale": all(n["in_scale"] for n in notes), "time": t}


def make_session(events, scale="C major", bpm=60.0) -> dict:
    return {"meta": {"scale": scale, "bpm": bpm, "date": "2026-01-01T12:00:00"},
            "events": events}


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def steady_session():
    """Eight in-scale quarter notes at 60 BPM, exactly one second apart."""
    return make_session([event(float(i), [60 + [0, 2, 4, 5, 7, 9, 11, 12][i]])
                         for i in range(8)])
