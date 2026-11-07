# The real Ollama/Azure APIs are never called: requests.post is blocked in
# conftest.py and replaced here by fakes that record what would be sent.

import json

import pytest
import requests

import agent
from conftest import write_json

FEEDBACK = {
    "meta": {"scale": "C major", "bpm": 60, "date": "2026-01-01"},
    "note_accuracy": {"score": 92, "total_notes": 100, "in_scale_count": 92,
                      "in_scale_ratio": 0.92, "assessment": "Good: few notes out of scale",
                      "frequent_errors": [{"note": "C#", "count": 5}], "avg_velocity": 64.0},
    "rhythm": {"timing_score": 70, "timing_assessment": "Mostly steady", "hesitation_count": 3,
               "dominant_values": [{"value": "quarter", "count": 40}]},
    "harmony": {"status": "no_data"},
    "suggestions": [{"priority": 2, "area": "rhythm", "message": "Keep the flow."}],
}


class FakeResponse:
    def __init__(self, status=200, content="Hello learner", body=None):
        self.status_code = status
        self._body = body if body is not None else {"choices": [{"message": {"content": content}}]}
        self.text = json.dumps(self._body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


@pytest.fixture
def sent(monkeypatch):
    """Replace requests.post with a fake; returns the list of recorded calls."""
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()
    monkeypatch.setattr(requests, "post", fake_post)
    return calls


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

def test_system_prompt_contains_session_data():
    prompt = agent.build_system_prompt(FEEDBACK, "en")
    assert prompt.startswith(agent.SYSTEM_PROMPT_EN)
    for expected in ["Key    : C major", "Good: few notes out of scale",
                     "Out-of-key notes: C#", "Avg loudness    : 64.0", "Mostly steady",
                     "[RHYTHM] Keep the flow.", "--- CORRECTION CONTEXT ---"]:
        assert expected in prompt
    assert "EXPRESSIVE ANALYSIS (main focus) ---" not in prompt


def test_system_prompt_reads_legacy_misspelled_keys():
    legacy = json.loads(json.dumps(FEEDBACK))
    legacy["note_accuracy"]["assessement"] = legacy["note_accuracy"].pop("assessment")
    legacy["rhythm"]["timing_assessement"] = legacy["rhythm"].pop("timing_assessment")
    prompt = agent.build_system_prompt(legacy, "en")
    assert "Good: few notes out of scale" in prompt and "Mostly steady" in prompt


def test_system_prompt_french():
    assert agent.build_system_prompt(FEEDBACK, "fr").startswith(agent.SYSTEM_PROMPT_FR)


@pytest.mark.parametrize("score,phrase", [(90, "give me feedback"),
                                          (60, "some good moments"),
                                          (30, "focus on first")])
def test_opening_depends_on_score(score, phrase):
    fb = {"note_accuracy": {"score": score}}
    assert phrase in agent.generate_opening(fb, "en")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

def test_ollama_config(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = agent.load_config("ollama", "llama3.1", 300)
    assert cfg["url"] == "http://localhost:11434/v1/chat/completions"
    assert cfg["model"] == "llama3.1" and cfg["max_tokens"] == 300
    assert "api-key" not in cfg["headers"]


def test_ollama_host_without_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.5:11434")
    assert agent.load_config("ollama")["url"] == "http://192.168.1.5:11434/v1/chat/completions"


def test_azure_config(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://lab.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-test")
    cfg = agent.load_config("azure")
    assert cfg["url"].startswith("https://lab.openai.azure.com/openai/deployments/gpt-test/")
    assert cfg["headers"]["api-key"] == "secret"


def test_azure_config_missing_env_exits(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
    with pytest.raises(SystemExit):
        agent.load_config("azure")


# ─────────────────────────────────────────────
# API calls (faked)
# ─────────────────────────────────────────────

def test_call_model_ollama_payload(sent):
    cfg = agent.load_config("ollama", "mistral", 123)
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert agent.call_model(cfg, "SYS", history, "question") == "Hello learner"
    payload = sent[0]["json"]
    assert payload["model"] == "mistral" and payload["max_tokens"] == 123
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert payload["messages"][-1]["content"] == "question"


def test_call_model_azure_payload_has_no_model_field(sent, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://lab.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "secret")
    agent.call_model(agent.load_config("azure"), "SYS", [], "q")
    assert "model" not in sent[0]["json"]
    assert sent[0]["headers"]["api-key"] == "secret"


def test_call_model_connection_error_hint(monkeypatch):
    def down(*a, **kw):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(requests, "post", down)
    out = agent.call_model(agent.load_config("ollama"), "SYS", [], "q")
    assert out.startswith("[ERROR]") and "ollama serve" in out


def test_call_model_unknown_model_hint(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse(
        404, body={"error": {"message": "model 'x' not found"}}))
    out = agent.call_model(agent.load_config("ollama", "x"), "SYS", [], "q")
    assert "model 'x' not found" in out and "ollama pull x" in out


def test_call_model_unexpected_format(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse(body={"oops": 1}))
    assert agent.call_model(agent.load_config("ollama"), "SYS", [], "q").startswith("[ERROR]")


# ─────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────

def test_dry_run_makes_no_call(tmp_path, capsys):
    path = write_json(tmp_path / "feedback_s.json", FEEDBACK)
    agent.main([str(path), "--dry-run"])        # conftest blocks requests.post
    out = capsys.readouterr().out
    assert "--- CORRECTION CONTEXT ---" in out and "--- OPENING MESSAGE ---" in out


def test_conversation_keeps_errors_out_of_history(tmp_path, monkeypatch):
    path = write_json(tmp_path / "feedback_s.json", FEEDBACK)
    replies = iter([FakeResponse(content="Opening answer"),
                    FakeResponse(500, body={"error": {"message": "boom"}}),
                    FakeResponse(content="Second answer")])
    monkeypatch.setattr(requests, "post", lambda *a, **kw: next(replies))
    inputs = iter(["", "first question", "second question", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    agent.main([str(path), "--model", "mistral"])

    log = json.loads((tmp_path / "conversation_feedback_s.json").read_text())
    assert log["provider"] == "ollama" and log["model"] == "mistral"
    assert [m["content"] for m in log["history"]] == [
        agent.generate_opening(FEEDBACK, "en"), "Opening answer",
        "second question", "Second answer"]


def test_opening_error_exits(tmp_path, monkeypatch):
    path = write_json(tmp_path / "feedback_s.json", FEEDBACK)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse(500, body={}))
    with pytest.raises(SystemExit):
        agent.main([str(path)])


# ─────────────────────────────────────────────
# Expressive analysis
# ─────────────────────────────────────────────

EXPRESSIVE = {
    "emotion": "sad", "target": "Q3", "target_name": "sad",
    "heard_as": "Q1", "heard_as_name": "happy / excited",
    "probabilities": {"Q1": 0.39, "Q2": 0.12, "Q3": 0.21, "Q4": 0.28},
    "advice": [{"cue": "velocity_mean", "name": "Loudness", "value": 69.4, "target": 54.6,
                "gap": 1.04, "advice": "Play softer, with a lighter touch"}],
    "already_there": [{"name": "Timing freedom"}],
    "what_if": {"cues_changed": ["velocity_mean"],
                "target_probability": {"before": 0.21, "after": 0.36}},
    "fixed_by_piece": [{"name": "Mode (major / minor)"}, {"name": "Register"}],
    "model": "EMOPIA 2.2 (1071 clips, 387 songs)",
}


def test_prompt_puts_expression_first():
    prompt = agent.build_system_prompt(FEEDBACK, "en", EXPRESSIVE)
    assert prompt.index("EXPRESSIVE ANALYSIS") < prompt.index("CORRECTION CONTEXT")
    for expected in ["Learner's intention   : sad", "happy / excited (39%)",
                     "1. Play softer, with a lighter touch (Loudness: currently clearly "
                     "louder than typical sad performances)", "from 21% to 36%",
                     "Already matching the intention: Timing freedom",
                     "never ask to change): Mode (major / minor), Register"]:
        assert expected in prompt


def test_opening_states_the_intention():
    assert "sound sad" in agent.generate_opening(FEEDBACK, "en", EXPRESSIVE)
    assert "« sad »" in agent.generate_opening(FEEDBACK, "fr", EXPRESSIVE)


def test_expressive_analysis_found_next_to_feedback(tmp_path):
    feedback = write_json(tmp_path / "feedback_session_x.json", FEEDBACK)
    assert agent.load_expressive(feedback) is None
    write_json(tmp_path / "expressive_session_x.json", EXPRESSIVE)
    assert agent.load_expressive(feedback)["emotion"] == "sad"


def test_conversation_log_records_intention(tmp_path, monkeypatch):
    path = write_json(tmp_path / "feedback_s.json", FEEDBACK)
    write_json(tmp_path / "expressive_s.json", EXPRESSIVE)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse(content="ok"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    agent.main([str(path)])
    log = json.loads((tmp_path / "conversation_feedback_s.json").read_text())
    assert log["intention"] == "sad" and "sound sad" in log["history"][0]["content"]


@pytest.mark.parametrize("cue,gap,expected", [
    ("velocity_mean", 0.6, "slightly louder than typical sad performances"),
    ("tempo_bpm", -1.5, "clearly slower than typical sad performances"),
    ("articulation", -2.5, "much more detached than typical sad performances"),
])
def test_gaps_are_described_in_words(cue, gap, expected):
    assert agent.describe_gap(cue, gap, "sad") == expected
