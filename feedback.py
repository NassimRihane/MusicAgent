# feedback.py
#
# Step 2 — Pedagogical feedback interpreter
#
# Reads a session JSON (from transcription.py) and optionally a chord
# analysis JSON (from chord_analysis.py), then produces structured
# feedback organised into three levels:
#
#   LEVEL 1 — Note accuracy    : are the right notes played?
#   LEVEL 2 — Rhythm           : are durations correct?
#   LEVEL 3 — Harmony          : do the chords fit the scale/progression?
#
# Output:
#   feedback_<session>.json    — machine-readable feedback for the agent
#   feedback_<session>.txt     — human-readable summary
#
# Usage:
#   python feedback.py sessions/session_20260101_120000.json
#   python feedback.py sessions/session_20260101_120000.json \
#                      --chords sessions/session_20260101_120000_chords.json


import argparse
import json
import sys
from pathlib import Path
from collections import Counter



# ─────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Pedagogical feedback interpreter")
parser.add_argument("session", help="Path to a session JSON file (from transcription.py)")
parser.add_argument("--chords", default=None, help="Path to a chord analysis JSON (from chord_analysis.py) - optional")
parser.add_argument("--reference", default=None, help="Path to a reference session JSON to compare against - optional")
args = parser.parse_args()


# ─────────────────────────────────────────────
# 1. MUSIC THEORY HELPERS
# ─────────────────────────────────────────────


NOTES_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', "B"]

NOTES_NAMES_TO_PC = {'C': 0,'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, "B": 11}

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10]
}

DIATONIC_CHORDS = {
    "major": {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"},
    "minor": {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}
}

# Onsets closer than this are treated as simultaneous (one chord)
SIMULTANEOUS_SEC = 0.05
# Minimum number of inter-onset intervals needed to score timing
MIN_IOIS = 3

rhythm_BEAT_VALUE = {
    "whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "16th": 0.25,
    "dotted-half": 3.0, "dotted-quarter": 1.5, "dotted-eighth": 0.75,
    "quarter-triplet": 2/3, "eighth-triplet": 1/3,
}



def build_scale(root_name: str, scale_type: str) -> set:
    root_pc = NOTES_NAMES_TO_PC.get(root_name, 0)
    intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])
    return set((root_pc + i) % 12 for i in intervals)

def note_pc(note_name: str) -> int:
    base = ''.join(c for c in note_name if not c.isdigit() and c != '-')
    return NOTES_NAMES_TO_PC.get(base,0)




# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)




# ─────────────────────────────────────────────
# 3. _NOTE ACCURACY ANALYSIS
# ─────────────────────────────────────────────

def analyse_notes(events: list, scale_pc: set) -> dict:
    """
    Evaluate note accuracy:
    - ration of notes in scale vs out of scale
    - most common out-of-scale notes (likely errors)
    - average loudness (MIDI velocity estimated by basic-pitch)
    - total note count
    """
    total_notes = 0
    in_scale_count = 0
    out_of_scale = []
    velocities = []
    out_pc_counter = Counter()

    for event in events:
        note_list = event.get("notes", [])
        for n in note_list:
            total_notes += 1
            pc = n.get("pitch_class", note_pc(n.get("name", "C")))
            if pc in scale_pc:
                in_scale_count +=1

            else:
                out_of_scale.append(n.get("name", "?"))
                out_pc_counter[NOTES_NAMES[pc]] += 1
            # Older sessions stored velocity/127 under the name "confidence"
            vel = n.get("velocity")
            if vel is None and "confidence" in n:
                vel = n["confidence"] * 127
            if vel is not None:
                velocities.append(vel)
    
    if total_notes == 0:
        return {"total_notes": 0, "in_scale_ratio": 0.0, "status": "no_data"}
    
    ratio = in_scale_count / total_notes
    avg_vel = sum(velocities)/len(velocities) if velocities else None

    # Get the 3 most frequent errors
    frequent_errors = [{"note": name, "count": count} for name, count in out_pc_counter.most_common(3)]

    score = round(ratio*100) # percentage

    # Feedback readable by user
    if ratio >= 0.95:
        assessment = "Excellent: almost all notes are in scale"
    elif ratio >= 0.8:
        assessment = "Good: few notes out of scale"
    elif ratio >= 0.6:
        assessment = "Mid: need a lot of correction"
    else:
        assessment = "Work again: too many false notes"

    return{
        "total_notes": total_notes,
        "in_scale_count": in_scale_count,
        "out_of_scale_count": total_notes - in_scale_count,
        "in_scale_ratio": round(ratio, 3),
        "score": score,
        "assessment": assessment,
        "frequent_errors": frequent_errors,
        "avg_velocity": round(avg_vel, 1) if avg_vel is not None else None
    }





# ─────────────────────────────────────────────
# 4. rhythm ANALYSIS
# ─────────────────────────────────────────────

def analyse_rhythm(events: list, bpm: float) -> dict:
    """
    Evaluate rhythm consistency:
        - distribution of note values played
        - average timing regularity (inter-onset intervals vs expected)
        - detection of hesitations (long gaps between notes)
    """
    if not events:
        return {"status": "no_data"}
    
    beat_duration = 60.0 / bpm # seconds per quarter (noires)
    rhythm_counts = Counter()
    raw_durations = []
    ioi_deviations = []  # inter onset interval
    hesitations = []

    prev_time = None
    for event in events:
        rhythm = event.get("rhythm")
        if rhythm:
            rhythm_counts[rhythm] += 1

        dur = event.get("duration")
        if dur is not None:
            raw_durations.append(dur)
        
        t = event.get("time", 0.0)
        if prev_time is not None:
            gap = t - prev_time
            if gap < SIMULTANEOUS_SEC:
                continue  # same onset (e.g. split chord), not a rhythmic interval
            if gap > beat_duration*2:
                hesitations.append({
                    "at": round(t, 2),
                    "gap_sec": round(gap, 2),
                    "gap_beats": round(gap/beat_duration, 1),
                })
            else:
                beats = gap/beat_duration
                nearest = min(rhythm_BEAT_VALUE.values(), key=lambda b: abs(b-beats))
                deviation = abs(beats - nearest)/nearest if nearest>0 else 0
                ioi_deviations.append(deviation)
        prev_time = t
    
    dominant_values = [{"value": v, "count": c} for v, c in rhythm_counts.most_common(4)]

    if len(ioi_deviations) < MIN_IOIS:
        # Too few measurable intervals: don't pretend the timing was steady
        return {
            "total_events": len(events),
            "dominant_values": dominant_values,
            "timing_assessment": "Not enough consecutive notes to judge timing",
            "hesitation_count": len(hesitations),
            "hesitations": hesitations[:5],
        }

    avg_deviation = round(sum(ioi_deviations) / len(ioi_deviations), 3)
    
    # Score penalise high deviation and hesitations
    timing_score = max(0, round((1 - avg_deviation)*100))

    if avg_deviation < 0.1:
        timing_assessment = "Steady rhythm"
    elif avg_deviation < 0.25:
        timing_assessment = "Mostly steady"
    else:
        timing_assessment = "rhythm too irregular"

    return{
        "total_events": len(events),
        "dominant_values": dominant_values,
        "avg_ioi_deviation": avg_deviation,
        "timing_score": timing_score,
        "timing_assessment": timing_assessment,
        "hesitation_count": len(hesitations),
        "hesitations": hesitations[:5]  # show only 5
    }



# ─────────────────────────────────────────────
# 5. HARMONY ANALYSIS
# ─────────────────────────────────────────────

def analyse_harmony(chord_events: list, scale_pc: set,
                    scale_root: str, scale_type: str) -> dict:
    """
    Evaluate harmonic coherence:
      - ratio of chords in scale
      - detection of non-diatonic chords
      - identification of common progressions (I-IV-V, ii-V-I, etc.)
    """
    if not chord_events:
        return {"status": "no_data",
                "note": "Run chord_analysis.py to enable harmony feedback."}
 
    # Ignore silence / unknown segments so they don't lower the score
    chord_events = [e for e in chord_events
                    if e.get("fit") not in ("no_chord", "unknown")
                    and e.get("chord", "N") not in ("N", "X")]
    if not chord_events:
        return {"status": "no_data", "note": "No chords detected in the recording."}

    total    = len(chord_events)
    in_scale = sum(1 for e in chord_events if e.get("fit") == "in_scale")
    partial  = sum(1 for e in chord_events if e.get("fit") == "partial")
    out      = sum(1 for e in chord_events if e.get("fit") == "out_of_scale")
 
    ratio = in_scale / total if total > 0 else 0.0
 
    # Extract chord root sequence for progression analysis
    chord_labels  = [e.get("chord", "N") for e in chord_events
                     if e.get("chord", "N") not in ("N", "X")]
    progressions  = _detect_progressions(chord_labels, scale_root, scale_type)
 
    if ratio >= 0.90:
        harmony_assessment = "Strong harmonic coherence: chords fit the scale well."
    elif ratio >= 0.70:
        harmony_assessment = "Good harmony: a few non-diatonic chords."
    elif ratio >= 0.50:
        harmony_assessment = "Moderate harmony: consider staying closer to the key."
    else:
        harmony_assessment = "Many out-of-key chords: review which chords belong to this scale."
 
    non_diatonic = [
        e["chord"] for e in chord_events
        if e.get("fit") == "out_of_scale" and e.get("chord") not in ("N", "X")
    ]
    non_diatonic_counts = Counter(non_diatonic).most_common(3)
 
    return {
        "total_chord_segments":  total,
        "in_scale_count":        in_scale,
        "partial_count":         partial,
        "out_of_scale_count":    out,
        "in_scale_ratio":        round(ratio, 3),
        "harmony_score":         round(ratio * 100),
        "harmony_assessment":    harmony_assessment,
        "non_diatonic_chords":   [{"chord": c, "count": n}
                                  for c, n in non_diatonic_counts],
        "progressions_detected": progressions,
    }


# Known progressions for chords
_KNOWN_PROGRESSIONS = {
    "I-IV-V": [0, 5, 7],
    "I-V-vi-IV": [0, 7, 9, 5],
    "ii-V-I": [2, 7, 0],
    "I-vi-IV-V": [0, 9, 5, 7],
    "vi-IV-I-V": [9, 5, 0, 7]
}


def _detect_progressions(chord_labels: list, root: str, scale_type: str) -> list:
    """
    Look for known chord progression
    """
    if not chord_labels:
        return []
    
    root_pc = NOTES_NAMES_TO_PC.get(root,0)

    # Convert chord labels to pitch class
    pcs = []
    for label in chord_labels:
        r = label.split(':')[0] if ':' in label else label
        pc = NOTES_NAMES_TO_PC.get(r)
        if pc is not None:
            pcs.append((pc - root_pc) % 12)
        
    found = []
    for name, pattern in _KNOWN_PROGRESSIONS.items():
        n = len(pattern)
        for i in range(len(pcs)- n + 1):
            if pcs[i:i+n] == pattern:
                found.append({"progression": name, "at_index": i})

    return found[:5]



# ─────────────────────────────────────────────
# 6. GLOBAL SCORE AND PRIORITY SUGGESTION
# ─────────────────────────────────────────────

def build_suggestions(note_fb: dict, rhythm_fb: dict, harmony_fb: dict, bpm: float) -> list:
    """
    Generate a prioritised list of actionable suggestions based on feedback
    Returned in order of priority
    """
    suggestions = []

    if note_fb.get("status") == "no_data":
        return [{
            "priority": 1,
            "area": "general",
            "message": "No notes were detected. Check the microphone input level and play again."
        }]

    note_score = note_fb.get("score", 100)
    timing_score = rhythm_fb.get("timing_score")
    harmony_score = harmony_fb.get("harmony_score", 100)


    # Note accuracy suggestions
    if note_score < 60:
        errors = note_fb.get("frequent_errors", [])
        error_names = ", ".join(e["note"] for e in errors[:2])
        suggestions.append({
            "priority": 1,
            "area": "notes",
            "message": f"Focus on note accuracy, especially {error_names}. "
                       f"Review which notes belong to the scale"
        })

    elif note_score < 80:
        suggestions.append({
            "priority": 2,
            "area": "notes",
            "message": "Good note accuracy overall, be careful about the occasional wrong notes"
        })
    

    # rhythm accuracy suggestions
    hesitations = rhythm_fb.get("hesitation_count", 0)
    if rhythm_fb.get("avg_ioi_deviation", 0) > 0.4:
        suggestions.append({
            "priority": 1,
            "area": "rhythm",
            "message":  "Work on rhythmic steadiness. Try practising with a metronome "
                        f"at {int(bpm * 0.7)} BPM (70% of current tempo) to build consistency."
        })
    
    elif hesitations > 2:
        suggestions.append({
            "priority": 2,
            "area":     "rhythm",
            "message":  f"There were {hesitations} noticeable pauses. "
                        "Try to keep the flow without stopping between phrases."
        })
 

    # Harmony accuracy suggestions
    if harmony_score < 50 and harmony_fb.get("status") != "no_data":
        non_diatonic = harmony_fb.get("non_diatonic_chords", [])
        if non_diatonic:
            chord_str = ", ".join(c["chord"] for c in non_diatonic[:2])
            suggestions.append({
                "priority": 2,
                "area":     "harmony",
                "message":  f"Some chords don't fit the scale ({chord_str}). "
                            "Review the diatonic chords for this key."
            })


    # Encouragements
    if note_score >= 90 and timing_score is not None and timing_score >= 80:
        suggestions.append({
            "priority": 3,
            "area": "encouragement",
            "message": "Good work! Both notes and rhythm are solid. "
                        "Try increasing the tempo or add nuances"
        })

    if not suggestions:
        suggestions.append({
            "priority": 3,
            "area": "general",
            "message": "Keep practicing regularly, consistency is key"
        })
    
    suggestions.sort(key=lambda s: s["priority"])
    return suggestions



# ─────────────────────────────────────────────
# 7. OUTPUT WRITERS
# ─────────────────────────────────────────────

def write_feedback_json(feedback: dict, base_path: Path) -> Path:
    out = base_path.parent / f"feedback_{base_path.stem}.json"
    with open(out, 'w', encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
    return out


def write_feedback_txt(feedback: dict, base_path: Path) -> Path:
    out = base_path.parent / f"feedback_{base_path.stem}.txt"
    meta = feedback["meta"]
    notes = feedback["note_accuracy"]
    rhythm = feedback["rhythm"]
    harmony = feedback["harmony"]
    sugg = feedback["suggestions"]

    lines = []
    lines.append("=" * 60)
    lines.append(" FEEDBACK REPORT ")
    lines.append(f"Session : {base_path.name}")
    lines.append(f"Scale : {meta.get('scale', '?')} | BPM: {meta.get('bpm', '?')}")
    lines.append("=" * 60)


    # Global score
    scores = [s for s in [notes.get("score"), rhythm.get("timing_score"), harmony.get("harmony_score")] if s is not None]
    global_score = round(sum(scores) / len(scores)) if scores else 0
    lines.append(f"\n GLOBAL SCORE : {global_score}/100\n")


    # Notes
    lines.append(" ── Level 1: Note accuracy ── ")
    if notes.get("status") == "no_data":
        lines.append(" No note data available")
    else:
        lines.append(f"Score : {notes.get('score',0)}/100")
        lines.append(f"Notes played: {notes.get('total_notes', 0)}")
        lines.append(f"In scale: {notes.get('in_scale_count', 0)} "
                    f"({round(notes.get('in_scale_ratio', 0)*100)}%)")
        lines.append(f"Assessment : {notes.get('assessment', '')}")
        if notes.get("frequent_errors"):
            errs = ", ".join(f"{e['note']} ({e['count']}x)" for e in notes["frequent_errors"])
            lines.append(f"Common errors: {errs}")



    # rhythm
    lines.append("")
    lines.append(" ── LEVEL 2 : Rhythm ── ")
    if rhythm.get("status") == "no_data":
        lines.append("  No rhythm data available.")
    else:
        lines.append(f"  Score       : {rhythm.get('timing_score', 'N/A')}/100")
        lines.append(f"  Assessment  : {rhythm.get('timing_assessment', '')}")
        if rhythm.get("hesitation_count", 0) > 0:
            lines.append(f"  Hesitations : {rhythm['hesitation_count']}")
        if rhythm.get("dominant_values"):
            vals = ", ".join(f"{v['value']} ({v['count']}×)"
                             for v in rhythm["dominant_values"])
            lines.append(f"  Main values : {vals}")



    # Harmony
    lines.append("")
    lines.append(" ── LEVEL 3 : Harmony ── ")
    if harmony.get("status") == "no_data":
        lines.append("  No chord data — run chord_analysis.py first.")
    else:
        lines.append(f"  Score       : {harmony.get('harmony_score', 0)}/100")
        lines.append(f"  Assessment  : {harmony.get('harmony_assessment', '')}")
        if harmony.get("non_diatonic_chords"):
            nd = ", ".join(f"{c['chord']} ({c['count']}×)"
                           for c in harmony["non_diatonic_chords"])
            lines.append(f"  Non-diatonic: {nd}")
        if harmony.get("progressions_detected"):
            progs = ", ".join(p["progression"]
                              for p in harmony["progressions_detected"])
            lines.append(f"  Progressions: {progs}")
    

    # Suggestions
    lines.append("")
    lines.append(" ── Suggestions ── ")
    for s in sugg:
        prefix = "  ★" if s["priority"] == 1 else "  ·"
        lines.append(f"{prefix} [{s['area'].upper()}] {s['message']}")
    
    lines.append("")
    lines.append("=" * 60)

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out




# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    session_path = Path(args.session)
    session      = load_json(str(session_path))
    meta         = session.get("meta", {})
    events       = session.get("events", [])
 
    scale_str  = meta.get("scale", "C major").split()
    scale_root = scale_str[0] if scale_str else "C"
    scale_type = scale_str[1] if len(scale_str) > 1 else "major"
    scale_pc   = build_scale(scale_root, scale_type)
    bpm        = float(meta.get("bpm", 60))
 
    # Load chord analysis if provided or auto-detected
    chord_events = []
    chord_path   = args.chords
    if args.reference:
        print("  Note: --reference is not implemented yet and is ignored.")
    if chord_path is None:
        auto = session_path.parent / f"{session_path.stem}_chords.json"
        if auto.exists():
            chord_path = str(auto)
            print(f"  Auto-detected chord analysis: {auto.name}")
 
    if chord_path:
        chord_data   = load_json(chord_path)
        chord_events = chord_data.get("chords", [])
 
    print("=" * 60)
    print("  Pedagogical Feedback")
    print(f"  Session  : {session_path.name}")
    print(f"  Scale    : {scale_root} {scale_type}  |  BPM: {bpm}")
    print(f"  Events   : {len(events)}")
    print(f"  Chords   : {len(chord_events)} segments")
    print("=" * 60)
 
    # Run all three analysis levels
    note_fb    = analyse_notes(events, scale_pc)
    rhythm_fb  = analyse_rhythm(events, bpm)
    harmony_fb = analyse_harmony(chord_events, scale_pc, scale_root, scale_type)
    suggestions = build_suggestions(note_fb, rhythm_fb, harmony_fb, bpm)
 
    feedback = {
        "meta":          meta,
        "note_accuracy": note_fb,
        "rhythm":        rhythm_fb,
        "harmony":       harmony_fb,
        "suggestions":   suggestions,
    }
 
    # Print summary
    scores = [s for s in [note_fb.get("score"), rhythm_fb.get("timing_score"),
                           harmony_fb.get("harmony_score")] if s is not None]
    global_score = round(sum(scores) / len(scores)) if scores else 0
    print(f"\n  Global score : {global_score}/100")
    print(f"  Notes        : {note_fb.get('assessment', note_fb.get('status', ''))}")
    print(f"  Rhythm       : {rhythm_fb.get('timing_assessment', rhythm_fb.get('status', ''))}")
    print(f"  Harmony      : {harmony_fb.get('harmony_assessment', harmony_fb.get('status', ''))}")
    print("\n  Suggestions:")
    for s in suggestions:
        prefix = "  ★" if s["priority"] == 1 else "  ·"
        print(f"{prefix} {s['message']}")
 
    # Save outputs
    json_out = write_feedback_json(feedback, session_path)
    txt_out  = write_feedback_txt(feedback, session_path)
    print(f"\n  JSON : {json_out}")
    print(f"  TXT  : {txt_out}")
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 