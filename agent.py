# agent.py
#
# Step 3 — Interactive feedback agent (Azure OpenAI)
#
# Uses your lab's Azure Enterprise Agreement — no personal billing.
#
# Configuration (set as environment variables before running):
#   export AZURE_OPENAI_ENDPOINT="https://your-lab.openai.azure.com/"
#   export AZURE_OPENAI_KEY="your-key-here"
#   export AZURE_OPENAI_DEPLOYMENT="model-exemple"   # your deployment name
#
# Find these in the Azure Portal:
#   Azure OpenAI resource → Keys and Endpoint
#
# Usage:
#   python agent.py sessions/feedback_session_20260101_120000.json
#   python agent.py sessions/feedback_session_20260101_120000.json --lang fr
#   python agent.py sessions/feedback_session_20260101_120000.json --dry-run
#       (prints the prompt that would be sent, without calling Azure)


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
# 0. ARGUMENTS
# ─────────────────────────────────────────────
 
parser = argparse.ArgumentParser(
    description="Interactive music feedback agent (Azure OpenAI)"
)
parser.add_argument("feedback",
    help="Path to a feedback JSON file (output of feedback.py)")
parser.add_argument("--lang", default="en", choices=["en", "fr"],
    help="Conversation language (en or fr, default: en)")
parser.add_argument("--max-tokens", type=int, default=800,
    help="Maximum tokens per response (default: 800)")
parser.add_argument("--dry-run", action="store_true",
    help="Print the system prompt and opening message without calling the API")
args = parser.parse_args()
 
 



# ─────────────────────────────────────────────
# 1. AZURE OPENAI CONFIGURATION
# ─────────────────────────────────────────────
 
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
 
    return {"url": url, "key": key, "deployment": deployment}



# ─────────────────────────────────────────────
# 2. SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────
 
SYSTEM_PROMPT_EN = """You are a warm, encouraging music teacher assistant helping a learner practise piano.
 
You have access to a structured analysis of their latest practice session.
Your role is to:
1. Explain clearly what went well and what needs improvement
2. Give concrete, actionable advice adapted to the learner's level
3. Motivate them to continue practising
4. Answer their questions about music theory when relevant
5. Suggest specific exercises when the data points to a recurring issue
 
Guidelines:
- Be encouraging before being critical
- Keep explanations concise — one idea per message
- Use musical terminology but explain it when the learner might not know it
- If the learner seems discouraged, prioritise motivation over technical detail
- Reference specific numbers from the session (e.g. "you played 87% of notes in scale")
- Suggest a concrete next step at the end of each response
 
The session analysis data is embedded below.
"""
 
SYSTEM_PROMPT_FR = """Tu es un assistant professeur de musique, chaleureux et encourageant, \
qui aide un apprenant à progresser au piano.
 
Tu disposes d'une analyse structurée de leur dernière session de pratique.
Ton rôle est de :
1. Expliquer clairement ce qui s'est bien passé et ce qui nécessite du travail
2. Donner des conseils concrets et actionnables adaptés au niveau de l'apprenant
3. Motiver l'apprenant à continuer
4. Répondre à ses questions de théorie musicale si besoin
5. Suggérer des exercices spécifiques quand les données révèlent un problème récurrent
 
Directives :
- Encourager avant de critiquer
- Une idée par message, explications concises
- Utiliser le vocabulaire musical mais l'expliquer si l'apprenant ne le connaît pas
- Si l'apprenant semble découragé, privilégier la motivation au détail technique
- Référencer les données chiffrées de la session (ex : "87 % de tes notes étaient dans la gamme")
- Proposer une prochaine étape concrète à la fin de chaque réponse
 
Les données d'analyse de la session sont intégrées ci-dessous.
"""


def _first(d: dict, *keys, default="no data"):
    """Return the first present key (older feedback files misspell some keys)."""
    for k in keys:
        if d.get(k):
            return d[k]
    return default


def build_system_prompt(feedback: dict, lang: str) -> str:
    """Build the system prompt with session analysis embedded."""
    base    = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    meta    = feedback.get("meta", {})
    notes   = feedback.get("note_accuracy", {})
    rhythm  = feedback.get("rhythm", {})
    harmony = feedback.get("harmony", {})
    sugg    = feedback.get("suggestions", [])
 
    scores = [s for s in [notes.get("score"), rhythm.get("timing_score"),
                           harmony.get("harmony_score")] if s is not None]
    global_score = round(sum(scores) / len(scores)) if scores else 0
 
    errors_str = ", ".join(e["note"] for e in notes.get("frequent_errors", [])[:3]) or "none"
    nd_str     = ", ".join(c["chord"] for c in harmony.get("non_diatonic_chords", [])[:3]) or "none"
    progs_str  = ", ".join(p["progression"] for p in harmony.get("progressions_detected", [])[:3]) or "none detected"
    vals_str   = ", ".join(v["value"] for v in rhythm.get("dominant_values", [])[:3])
    sugg_lines = "\n".join(f"  [{s['area'].upper()}] {s['message']}" for s in sugg[:3])
 
    data_block = f"""
--- SESSION DATA ---
Scale  : {meta.get("scale", "?")}
BPM    : {meta.get("bpm", "?")}
Date   : {meta.get("date", "?")}
 
GLOBAL SCORE : {global_score}/100
 
NOTE ACCURACY ({notes.get("score", "N/A")}/100):
  Total notes     : {notes.get("total_notes", 0)}
  In scale        : {notes.get("in_scale_count", 0)} ({round(notes.get("in_scale_ratio", 0)*100)}%)
  Assessment      : {_first(notes, "assessment", "assessement", "status")}
  Frequent errors : {errors_str}
  Avg loudness    : {notes.get("avg_velocity", "?")} (MIDI velocity, 0-127)
 
RHYTHM ({rhythm.get("timing_score", "N/A")}/100):
  Assessment      : {_first(rhythm, "timing_assessment", "timing_assessement", "status")}
  Hesitations     : {rhythm.get("hesitation_count", 0)}
  Main note values: {vals_str}
  Avg IOI deviation: {rhythm.get("avg_ioi_deviation", "?")}
 
HARMONY ({harmony.get("harmony_score", "N/A")}/100):
  Assessment      : {harmony.get("harmony_assessment", harmony.get("status", "no data"))}
  Non-diatonic    : {nd_str}
  Progressions    : {progs_str}
 
PRIORITY SUGGESTIONS:
{sugg_lines}
--- END SESSION DATA ---
"""
    return base + data_block




# ─────────────────────────────────────────────
# 3. AZURE OPENAI API CALL
# ─────────────────────────────────────────────
 
def call_model(config: dict, system_prompt: str,
               history: list, user_message: str) -> str:
    """
    Send a chat completion request to Azure OpenAI.
 
    The Azure OpenAI API mirrors the OpenAI format exactly, with two differences:
    - URL:  {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...
    - Auth: header "api-key" instead of "Authorization: Bearer ..."
 
    The conversation history is included in full on every call because the
    API is stateless — the model has no memory between requests.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})
 
    payload = {
        "messages":    messages,
        "max_tokens":  args.max_tokens,
        "temperature": 0.7,
    }
 
    headers = {
        "api-key":      config["key"],
        "Content-Type": "application/json",
    }
 
    try:
        resp = requests.post(config["url"], json=payload,
                             headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
 
    except requests.exceptions.ConnectionError:
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
        return (f"[ERROR] Azure API returned HTTP {resp.status_code}. "
                f"{detail or resp.text[:300]}")
    except (KeyError, IndexError) as e:
        return f"[ERROR] Unexpected response format: {e}"
    except Exception as e:
        return f"[ERROR] {e}"



# ─────────────────────────────────────────────
# 4. CONTEXT-AWARE OPENING MESSAGE
# ─────────────────────────────────────────────
 
def generate_opening(feedback: dict, lang: str) -> str:
    """
    Choose a first message appropriate to the session score.
    A good score → request for a summary.
    A low score  → ask what to prioritise.
    """
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
 
def main():
    feedback_path = Path(args.feedback)
    if not feedback_path.exists():
        print(f"ERROR: file not found — {feedback_path}")
        sys.exit(1)
 
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    lang     = args.lang
 
    system_prompt = build_system_prompt(feedback, lang)
    if args.dry_run:
        print(system_prompt)
        print("--- OPENING MESSAGE ---")
        print(generate_opening(feedback, lang))
        return

    config   = load_azure_config()
    history       = []   # grows with each turn; sent in full on every API call
 
    meta   = feedback.get("meta", {})
    scores = [s for s in [
        feedback.get("note_accuracy", {}).get("score"),
        feedback.get("rhythm", {}).get("timing_score"),
        feedback.get("harmony", {}).get("harmony_score"),
    ] if s is not None]
    global_score = round(sum(scores) / len(scores)) if scores else 0
 
    print("=" * 60)
    print("  Music Practice Agent  —  Azure OpenAI")
    print(f"  Session    : {feedback_path.stem}")
    print(f"  Scale      : {meta.get('scale', '?')}  |  BPM: {meta.get('bpm', '?')}")
    print(f"  Score      : {global_score}/100")
    print(f"  Deployment : {config['deployment']}")
    print(f"  Language   : {lang}")
    print("  Type 'quit' or press Ctrl+C to end.")
    print("=" * 60)
    print()
 
    # First turn — automatic context-aware opening
    opening = generate_opening(feedback, lang)
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
            "deployment": config["deployment"],
            "history":    history,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nConversation saved: {log_path}")
 
 
if __name__ == "__main__":
    main()
