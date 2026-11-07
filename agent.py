# agent.py
#
# Step 3 — Interactive feedback agent (Ollama or Azure OpenAI)
#
# Providers:
#   ollama (default) — local model, free. Requires the Ollama app running
#                      and the model pulled:  ollama pull mistral
#                      Optional: export OLLAMA_HOST="http://localhost:11434"
#   azure            — your lab's Azure OpenAI deployment. Set:
#                      export AZURE_OPENAI_ENDPOINT="https://your-lab.openai.azure.com/"
#                      export AZURE_OPENAI_KEY="your-key-here"
#                      export AZURE_OPENAI_DEPLOYMENT="model-exemple"
#
# The coach's main input is the expressive analysis (expressive_feedback.py),
# found automatically next to the feedback file; the correction feedback
# (feedback.py) is passed as context.
#
# Usage:
#   python agent.py sessions/feedback_session_20260101_120000.json
#   python agent.py sessions/feedback_session_20260101_120000.json --lang fr
#   python agent.py sessions/feedback_session_20260101_120000.json --model llama3.1
#   python agent.py sessions/feedback_session_20260101_120000.json --provider azure
#   python agent.py sessions/feedback_session_20260101_120000.json --dry-run
#       (prints the prompt that would be sent, without calling any model)


import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests is not installed. Run : pip install requests")
    sys.exit(1)



# ─────────────────────────────────────────────
# 0. MODEL SELECTION & ARGUMENTS
# ─────────────────────────────────────────────

# Change these two lines to switch model (or use --provider / --model).
PROVIDER = "ollama"     # "ollama" or "azure"
MODEL    = "mistral"    # Ollama model name (see: ollama list)
 
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive music feedback agent (Ollama or Azure OpenAI)"
    )
    parser.add_argument("feedback",
        help="Path to a feedback JSON file (output of feedback.py)")
    parser.add_argument("--lang", default="en", choices=["en", "fr"],
        help="Conversation language (en or fr, default: en)")
    parser.add_argument("--max-tokens", type=int, default=800,
        help="Maximum tokens per response (default: 800)")
    parser.add_argument("--expressive", default=None,
        help="Expressive analysis JSON (default: expressive_<session>.json next to the feedback)")
    parser.add_argument("--dry-run", action="store_true",
        help="Print the system prompt and opening message without calling the API")
    parser.add_argument("--provider", default=PROVIDER, choices=["ollama", "azure"],
        help=f"LLM provider (default: {PROVIDER})")
    parser.add_argument("--model", default=MODEL,
        help=f"Ollama model name (default: {MODEL}); ignored for azure")
    return parser.parse_args(argv)
 
 



# ─────────────────────────────────────────────
# 1. PROVIDER CONFIGURATION
# ─────────────────────────────────────────────

def load_config(provider: str, model: str = MODEL, max_tokens: int = 800) -> dict:
    """Return {provider, url, headers, model, timeout, max_tokens} for the chosen provider."""
    config = load_azure_config() if provider == "azure" else load_ollama_config(model)
    config["max_tokens"] = max_tokens
    return config


def load_ollama_config(model: str = MODEL) -> dict:
    """
    Ollama exposes an OpenAI-compatible endpoint, so the request format is the
    same as Azure's; only the URL and the 'model' field differ.
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    return {
        "provider": "ollama",
        "url":      f"{host}/v1/chat/completions",
        "headers":  {"Content-Type": "application/json"},
        "model":    model,
        "timeout":  180,   # local models can be slow, especially the first call
    }

 
def load_azure_config() -> dict:
    """
    Load Azure OpenAI credentials from environment variables.
    Exits with a clear error message if any required variable is missing.
    """

    endpoint    = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key         = os.environ.get("AZURE_OPENAI_KEY", "")
    deployment  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "model-exemple")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
 
    missing = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not key:
        missing.append("AZURE_OPENAI_KEY")
 
    if missing:
        print("ERROR: Missing Azure OpenAI configuration.\n")
        print("Set the following environment variables before running:\n")
        for var in missing:
            print(f"  export {var}=<value>")
        print("\nFind these values in the Azure Portal:")
        print("  Azure OpenAI resource → Keys and Endpoint\n")
        sys.exit(1)
 
    # Azure OpenAI chat completions endpoint format:
    # POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
    url = (f"{endpoint}/openai/deployments/{deployment}"
           f"/chat/completions?api-version={api_version}")
 
    return {
        "provider": "azure",
        "url":      url,
        "headers":  {"api-key": key, "Content-Type": "application/json"},
        "model":    deployment,
        "timeout":  30,
    }



# ─────────────────────────────────────────────
# 2. SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────
 
SYSTEM_PROMPT_EN = """You are a warm, encouraging piano teacher who helps learners play \
expressively: not only the right notes, but with the emotion they intend.

You receive two analyses of the learner's latest performance:
- an EXPRESSIVE ANALYSIS (the main focus): the emotion a listener model hears,
  the learner's intention, and the playing changes that would bring the
  performance closer to that intention, most important first;
- a CORRECTION CONTEXT: wrong notes, rhythm and harmony.

How to coach:
- Start from the intention: say what already conveys it, then give the one or
  two most important changes from the expressive analysis.
- Turn measurements into musical language a learner can act on (touch, weight,
  pedal, breathing, phrasing). Use at most one number per idea.
- The intention comes first. Something the correction context calls an error
  (for example irregular timing) can be a deliberate expressive choice, such as
  rubato in a sad piece: never ask the learner to remove what the expressive
  analysis asks for.
- Mention a correction only if it really gets in the way of the expression.
- Never suggest changing what is set by the piece (its key, register or texture).
- Keep it short: one idea per message, and end with one concrete thing to try.

The analysis data is embedded below.
"""
 
SYSTEM_PROMPT_FR = """Tu es un professeur de piano chaleureux et encourageant, qui aide les \
apprenants à jouer de manière expressive : pas seulement les bonnes notes, mais avec \
l'émotion qu'ils veulent transmettre.

Tu reçois deux analyses de la dernière interprétation de l'apprenant :
- une ANALYSE EXPRESSIVE (le sujet principal) : l'émotion qu'un modèle d'écoute
  perçoit, l'intention de l'apprenant, et les changements de jeu qui
  rapprocheraient l'interprétation de cette intention, du plus important au moins important ;
- un CONTEXTE DE CORRECTION : fausses notes, rythme et harmonie.

Comment accompagner :
- Pars de l'intention : dis ce qui la transmet déjà, puis donne les un ou deux
  changements les plus importants de l'analyse expressive.
- Traduis les mesures en langage musical concret (toucher, poids, pédale,
  respiration, phrasé). Au plus un chiffre par idée.
- L'intention passe en premier. Ce que le contexte de correction appelle une
  erreur (par exemple un rythme irrégulier) peut être un choix expressif, comme
  le rubato dans un morceau triste : ne demande jamais de supprimer ce que
  l'analyse expressive demande.
- Ne mentionne une correction que si elle gêne vraiment l'expression.
- Ne propose jamais de changer ce qui est fixé par le morceau (tonalité, registre, texture).
- Reste bref : une idée par message, et termine par une chose concrète à essayer.

Les données d'analyse sont intégrées ci-dessous.
"""


def _first(d: dict, *keys, default="no data"):
    """Return the first present key (older feedback files misspell some keys)."""
    for k in keys:
        if d.get(k):
            return d[k]
    return default


# How a gap reads in words: (word when the learner is above the target, below it)
GAP_WORDS = {
    "tempo_bpm":          ("faster", "slower"),
    "velocity_mean":      ("louder", "softer"),
    "velocity_std":       ("more contrasted in dynamics", "more even in dynamics"),
    "articulation":       ("more connected (legato)", "more detached"),
    "timing_variability": ("freer in timing", "steadier in timing"),
}


def describe_gap(cue: str, gap: float, target_name: str) -> str:
    """'clearly louder than typical sad performances' (gap in inter-quartile ranges)."""
    size = "slightly" if abs(gap) < 1 else "clearly" if abs(gap) < 2 else "much"
    above, below = GAP_WORDS.get(cue, ("higher", "lower"))
    return f"{size} {above if gap > 0 else below} than typical {target_name} performances"


def expressive_block(x: dict) -> str:
    """Text block for the expressive analysis (output of expressive_feedback.py)."""
    p = x["probabilities"]
    target = x["target"]
    lines = [
        "--- EXPRESSIVE ANALYSIS (main focus) ---",
        f"Learner's intention   : {x.get('emotion', x['target_name'])} ({target}, {x['target_name']})",
        f"What the model hears  : {x['heard_as']} {x['heard_as_name']} ({p[x['heard_as']]:.0%}); "
        f"the intention is at {p[target]:.0%}",
        f"Listener model        : Random Forest trained on {x['model']}",
        "Priority changes (most important first):",
    ]
    for i, c in enumerate(x["advice"], 1):
        lines.append(f"  {i}. {c['advice']} ({c['name']}: currently "
                     f"{describe_gap(c['cue'], c['gap'], x['target_name'])})")
    if not x["advice"]:
        lines.append("  none: the controllable cues already match the intention")
    w = x["what_if"]["target_probability"]
    if x["advice"]:
        lines.append(f"With these changes the model's probability for the intention would go "
                     f"from {w['before']:.0%} to {w['after']:.0%}.")
    if x["already_there"]:
        lines.append("Already matching the intention: "
                     + ", ".join(c["name"] for c in x["already_there"]))
    lines.append("Set by the piece (never ask to change): "
                 + ", ".join(c["name"] for c in x["fixed_by_piece"]))
    lines.append("--- END EXPRESSIVE ANALYSIS ---")
    return "\n".join(lines)


def build_system_prompt(feedback: dict, lang: str, expressive: dict = None) -> str:
    """Build the system prompt with the session analyses embedded."""
    base    = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    meta    = feedback.get("meta", {})
    notes   = feedback.get("note_accuracy", {})
    rhythm  = feedback.get("rhythm", {})
    harmony = feedback.get("harmony", {})
    sugg    = feedback.get("suggestions", [])

    errors_str = ", ".join(e["note"] for e in notes.get("frequent_errors", [])[:3]) or "none"
    nd_str     = ", ".join(c["chord"] for c in harmony.get("non_diatonic_chords", [])[:3]) or "none"
    progs_str  = ", ".join(p["progression"] for p in harmony.get("progressions_detected", [])[:3]) or "none detected"
    sugg_lines = "\n".join(f"  [{s['area'].upper()}] {s['message']}" for s in sugg[:3])

    correction = f"""
--- CORRECTION CONTEXT ---
Key    : {meta.get("scale", "?")} ({meta.get("scale_source", "declared")})
Tempo  : {rhythm.get("tempo_bpm", meta.get("bpm", "?"))} BPM
Date   : {meta.get("date", "?")}

NOTES ({notes.get("score", "N/A")}/100):
  Total notes     : {notes.get("total_notes", 0)}
  In key          : {notes.get("in_scale_count", 0)} ({round(notes.get("in_scale_ratio", 0)*100)}%)
  Assessment      : {_first(notes, "assessment", "assessement", "status")}
  Out-of-key notes: {errors_str}
  Avg loudness    : {notes.get("avg_velocity", "?")} (MIDI velocity, 0-127)

RHYTHM ({rhythm.get("timing_score", "N/A")}/100):
  Assessment      : {_first(rhythm, "timing_assessment", "timing_assessement", "status")}
  Hesitations     : {rhythm.get("hesitation_count", 0)}

HARMONY ({harmony.get("harmony_score", "N/A")}/100):
  Assessment      : {harmony.get("harmony_assessment", harmony.get("status", "no data"))}
  Progression     : {harmony.get("progression", "?")}
  Out-of-key      : {nd_str}
  Known patterns  : {progs_str}

CORRECTION SUGGESTIONS:
{sugg_lines}
--- END CORRECTION CONTEXT ---
"""
    if expressive:
        return base + "\n" + expressive_block(expressive) + "\n" + correction
    return base + correction



# ─────────────────────────────────────────────
# 3. AZURE OPENAI API CALL
# ─────────────────────────────────────────────
 
def call_model(config: dict, system_prompt: str,
               history: list, user_message: str) -> str:
    """
    Send a chat completion request (OpenAI format) to Ollama or Azure OpenAI.
 
    Both accept the OpenAI chat format; they differ only by:
    - URL:   Ollama {host}/v1/chat/completions
             Azure  {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...
    - Auth:  Azure needs an "api-key" header; Ollama needs none
    - Model: Ollama needs a "model" field; Azure takes it from the URL
 
    The conversation history is included in full on every call because the
    API is stateless — the model has no memory between requests.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})
 
    payload = {
        "messages":    messages,
        "max_tokens":  config.get("max_tokens", 800),
        "temperature": 0.7,
    }
    if config["provider"] == "ollama":
        payload["model"] = config["model"]
 
    try:
        resp = requests.post(config["url"], json=payload,
                             headers=config["headers"], timeout=config["timeout"])
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
 
    except requests.exceptions.ConnectionError:
        if config["provider"] == "ollama":
            return ("[ERROR] Cannot reach Ollama. Start the Ollama app "
                    "(or run: ollama serve) and check OLLAMA_HOST.")
        return ("[ERROR] Cannot reach Azure OpenAI. "
                "Check AZURE_OPENAI_ENDPOINT and your internet connection.")
    except requests.exceptions.Timeout:
        return "[ERROR] Request timed out — try again."
    except requests.exceptions.HTTPError:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            pass
        hint = ""
        if config["provider"] == "ollama" and resp.status_code == 404:
            hint = f" (Is the model pulled? Run: ollama pull {config['model']})"
        return (f"[ERROR] {config['provider']} API returned HTTP {resp.status_code}. "
                f"{detail or resp.text[:300]}{hint}")
    except (KeyError, IndexError) as e:
        return f"[ERROR] Unexpected response format: {e}"
    except Exception as e:
        return f"[ERROR] {e}"



# ─────────────────────────────────────────────
# 4. CONTEXT-AWARE OPENING MESSAGE
# ─────────────────────────────────────────────
 
def generate_opening(feedback: dict, lang: str, expressive: dict = None) -> str:
    """
    First message of the learner. With an expressive analysis: state the
    intention. Otherwise, choose by score: a good score asks for a summary,
    a low one asks what to prioritise.
    """
    if expressive:
        emotion = expressive.get("emotion", expressive["target_name"])
        if lang == "fr":
            return (f"Je voulais que ce morceau sonne « {emotion} ». "
                    "Comment est-ce que ça sonne, et que dois-je changer ?")
        return f"I wanted this piece to sound {emotion}. How did it sound, and what should I change?"
    scores = [s for s in [
        feedback.get("note_accuracy", {}).get("score"),
        feedback.get("rhythm", {}).get("timing_score"),
        feedback.get("harmony", {}).get("harmony_score"),
    ] if s is not None]
    global_score = round(sum(scores) / len(scores)) if scores else 50
 
    if lang == "fr":
        if global_score >= 80:
            return "J'ai terminé ma session. Donne-moi un retour sur comment ça s'est passé."
        elif global_score >= 55:
            return ("J'ai terminé ma session. Il y avait des bonnes choses et des choses "
                    "à améliorer. Qu'est-ce que tu en penses ?")
        else:
            return ("J'ai terminé ma session mais je pense que ça ne s'est pas très bien "
                    "passé. Tu peux m'expliquer ce que je dois travailler en priorité ?")
    else:
        if global_score >= 80:
            return "I just finished my practice session. Can you give me feedback on how it went?"
        elif global_score >= 55:
            return ("I finished my session. I think there were some good moments and "
                    "some things to improve. What do you think?")
        else:
            return ("I finished my session but I don't think it went very well. "
                    "Can you tell me what I should focus on first?")
 
 
 
# ─────────────────────────────────────────────
# 5. MAIN CONVERSATION LOOP
# ─────────────────────────────────────────────
 
def load_expressive(feedback_path: Path, explicit: str = None):
    """The expressive analysis for this session, if there is one."""
    path = Path(explicit) if explicit else feedback_path.with_name(
        feedback_path.name.replace("feedback_", "expressive_", 1))
    if path.exists() and path != feedback_path:
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main(argv=None):
    args = parse_args(argv)
    feedback_path = Path(args.feedback)
    if not feedback_path.exists():
        print(f"ERROR: file not found — {feedback_path}")
        sys.exit(1)
 
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    lang     = args.lang
    expressive = load_expressive(feedback_path, args.expressive)
 
    system_prompt = build_system_prompt(feedback, lang, expressive)
    if args.dry_run:
        print(system_prompt)
        print("--- OPENING MESSAGE ---")
        print(generate_opening(feedback, lang, expressive))
        return

    config   = load_config(args.provider, args.model, args.max_tokens)
    history       = []   # grows with each turn; sent in full on every API call
 
    meta   = feedback.get("meta", {})
    scores = [s for s in [
        feedback.get("note_accuracy", {}).get("score"),
        feedback.get("rhythm", {}).get("timing_score"),
        feedback.get("harmony", {}).get("harmony_score"),
    ] if s is not None]
    global_score = round(sum(scores) / len(scores)) if scores else 0
 
    print("=" * 60)
    print(f"  Music Practice Agent  —  {config['provider']}")
    print(f"  Session    : {feedback_path.stem}")
    print(f"  Scale      : {meta.get('scale', '?')}  |  BPM: {meta.get('bpm', '?')}")
    print(f"  Score      : {global_score}/100")
    if expressive:
        print(f"  Intention  : {expressive.get('emotion')}  |  heard as: "
              f"{expressive['heard_as_name']}")
    print(f"  Model      : {config['model']}")
    print(f"  Language   : {lang}")
    print("  Type 'quit' or press Ctrl+C to end.")
    print("=" * 60)
    print()
 
    # First turn — automatic context-aware opening
    opening = generate_opening(feedback, lang, expressive)
    you_label = "Vous" if lang == "fr" else "You"
    print(f"{you_label}: {opening}\n")
 
    response = call_model(config, system_prompt, history, opening)
    print(f"Agent: {response}\n")
    if response.startswith("[ERROR]"):
        sys.exit(1)
    history.append({"role": "user",      "content": opening})
    history.append({"role": "assistant", "content": response})
 
    # Conversation loop
    prompt_str = "Vous : " if lang == "fr" else "You  : "
    while True:
        try:
            user_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
 
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "au revoir", "q"):
            farewell = ("À bientôt ! Continue à pratiquer régulièrement." if lang == "fr"
                        else "Goodbye! Keep practising regularly.")
            print(f"\nAgent: {farewell}")
            break
 
        response = call_model(config, system_prompt, history, user_input)
        print(f"\nAgent: {response}\n")
        if response.startswith("[ERROR]"):
            continue  # keep errors out of the history sent to the model
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})
 
    # Save full conversation log
    log_path = feedback_path.parent / f"conversation_{feedback_path.stem}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta":       meta,
            "language":   lang,
            "intention":  expressive.get("emotion") if expressive else None,
            "provider":   config["provider"],
            "model":      config["model"],
            "history":    history,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nConversation saved: {log_path}")
 
 
if __name__ == "__main__":
    main()
