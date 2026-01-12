import streamlit as st
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# =========================================================
# CONFIG
# =========================================================

APP_TITLE = "Lens Readout"
DEFAULT_LENSES = ["relationship", "finance", "goals"]  # rename as you want

# IMPORTANT: use the *embed* URL, not forms.gle
GOOGLE_FORM_EMBED_URL = "<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSciFAyvy7W5cwepglPL_zrC0PfoQK951sbuO_1eMCPfZ6fW9w/viewform?embedded=true" width="640" height="1118" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>"  # e.g. https://docs.google.com/forms/d/e/.../viewform?embedded=true

# ---------------------------------------------------------
# Question Bank (YOU will paste your question dicts here)
# Each question dict format:
# {
#   "id": "r01",
#   "text": "Question text ...",
#   "var": "Trust",                # variable bucket name used in scoring
#   "w": 1.0,                      # weight of this question
#   "invert": False                # True if higher answer means worse
# }
# ---------------------------------------------------------
QUESTION_BANK: Dict[str, List[Dict[str, Any]]] = {
    "relationship": [],
    "finance": [],
    "goals": [],
}

# Follow-up bank is optional; if empty, app will auto-generate follow-ups from weakest vars
FOLLOWUP_BANK: Dict[str, List[Dict[str, Any]]] = {
    "relationship": [],
    "finance": [],
    "goals": [],
}

# Answer scale (change if you want)
MIN_SCORE = 1
MAX_SCORE = 5

# =========================================================
# HELPERS
# =========================================================

def ensure_state():
    defaults = {
        "mode": "single",                     # single | trilens
        "lens": DEFAULT_LENSES[0],
        "stage": "home",                      # home | questions | followups | export_form | results2 | trilens_results
        "run_id": None,

        "active_questions": [],
        "answers": {},                        # qid -> int
        "idx": 0,

        "followup_questions": [],
        "followup_answers": {},               # qid -> int
        "followup_idx": 0,

        "last_results": None,                 # store results2 payload
        "trilens_queue": [],                  # list of lens names to run
        "trilens_results": {},                # lens -> results payload
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_single_run(keep_lens=True):
    lens = st.session_state.lens
    st.session_state.active_questions = []
    st.session_state.answers = {}
    st.session_state.idx = 0
    st.session_state.followup_questions = []
    st.session_state.followup_answers = {}
    st.session_state.followup_idx = 0
    st.session_state.last_results = None
    if keep_lens:
        st.session_state.lens = lens


def sample_questions(lens: str, k: int = 25) -> List[Dict[str, Any]]:
    bank = QUESTION_BANK.get(lens, [])[:]
    if not bank:
        return []
    random.shuffle(bank)
    k = min(k, len(bank))
    return random.sample(bank, k=k)


def clamp_int(v, lo=MIN_SCORE, hi=MAX_SCORE):
    try:
        iv = int(v)
    except Exception:
        return lo
    return max(lo, min(hi, iv))


def score_run(lens: str, questions: List[Dict[str, Any]], answers: Dict[str, int]):
    """
    Returns:
      overall_pct: float (0..100)
      per_var: dict var -> {"pct": float, "raw": float, "max": float, "n": int}
      scored_rows: list of rows for debugging
      targets: list of weakest vars (for followups)
    """
    per_var_raw = {}
    per_var_max = {}
    per_var_n = {}
    rows = []

    for q in questions:
        qid = q["id"]
        if qid not in answers:
            continue

        var = q.get("var", "General")
        w = float(q.get("w", 1.0))
        invert = bool(q.get("invert", False))
        a = clamp_int(answers[qid])

        # normalize to 0..1
        norm = (a - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        if invert:
            norm = 1.0 - norm

        # weighted contribution
        raw = norm * w
        mx = 1.0 * w

        per_var_raw[var] = per_var_raw.get(var, 0.0) + raw
        per_var_max[var] = per_var_max.get(var, 0.0) + mx
        per_var_n[var] = per_var_n.get(var, 0) + 1

        rows.append({"id": qid, "var": var, "w": w, "invert": invert, "answer": a, "norm": norm})

    # compute per var pct + overall
    per_var = {}
    total_raw = 0.0
    total_max = 0.0

    for var in per_var_max:
        raw = per_var_raw.get(var, 0.0)
        mx = per_var_max.get(var, 0.0)
        pct = 0.0 if mx == 0 else (raw / mx) * 100.0
        per_var[var] = {"pct": pct, "raw": raw, "max": mx, "n": per_var_n.get(var, 0)}
        total_raw += raw
        total_max += mx

    overall_pct = 0.0 if total_max == 0 else (total_raw / total_max) * 100.0

    # targets = weakest 2 vars
    sorted_vars = sorted(per_var.items(), key=lambda kv: kv[1]["pct"])
    targets = [v for v, _ in sorted_vars[:2]]

    return overall_pct, per_var, rows, targets


def pick_followups(lens: str, targets: List[str], already_asked_ids: set, n: int = 10):
    """
    Prefer FOLLOWUP_BANK questions. If empty, auto-generate generic followups per target var.
    """
    bank = FOLLOWUP_BANK.get(lens, [])[:]
    picked = []

    if bank:
        candidates = [q for q in bank if q.get("var") in targets and q["id"] not in already_asked_ids]
        random.shuffle(candidates)
        picked = candidates[: min(n, len(candidates))]
    else:
        # auto-generate placeholder followups
        for i in range(n):
            var = targets[i % len(targets)] if targets else "General"
            qid = f"fu_{lens}_{var}_{i+1}"
            if qid in already_asked_ids:
                continue
            picked.append({
                "id": qid,
                "text": f"[AUTO FOLLOW-UP] Clarify {var}: What best describes you right now?",
                "var": var,
                "w": 1.0,
                "invert": False
            })

    return picked


def interpret(per_var: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Simple, readable narratives. You can expand this later.
    """
    if not per_var:
        return ["No scoring data yet. Add questions to your QUESTION_BANK."]

    # weakest + strongest vars
    weak = sorted(per_var.items(), key=lambda kv: kv[1]["pct"])[:2]
    strong = sorted(per_var.items(), key=lambda kv: kv[1]["pct"], reverse=True)[:2]

    lines = []
    lines.append(f"Top strengths: **{strong[0][0]}** ({strong[0][1]['pct']:.1f}%), **{strong[1][0]}** ({strong[1][1]['pct']:.1f}%).")
    lines.append(f"Biggest pressure points: **{weak[0][0]}** ({weak[0][1]['pct']:.1f}%), **{weak[1][0]}** ({weak[1][1]['pct']:.1f}%).")

    # “strain” logic: if one var is strong and another weak, it can create friction
    if strong and weak:
        lines.append(
            f"Pattern: your **{strong[0][0]}** is carrying weight while **{weak[0][0]}** is dragging. "
            f"That usually feels like progress in one area causing stress in another."
        )

    lines.append("First move: pick **one** weak variable and do a small, repeatable fix for 7 days (not a life overhaul).")
    return lines


def cross_lens_synthesis(trilens_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Build the “relationship + goals + finance” interplay story.
    """
    lines = []
    if not trilens_results:
        return ["No tri-lens results yet."]

    # Collect weakest var per lens + overall
    summary = []
    for lens, payload in trilens_results.items():
        overall = payload["overall"]
        per_var = payload["per_var"]
        weak = None
        if per_var:
            weak = min(per_var.items(), key=lambda kv: kv[1]["pct"])
        summary.append((lens, overall, weak))

    # Sort lenses by overall ascending (most trouble first)
    summary.sort(key=lambda x: x[1])

    # Narrative
    worst_lens = summary[0][0]
    lines.append(f"Most strained lens right now: **{worst_lens}** (lowest overall score).")

    for lens, overall, weak in summary:
        if weak:
            lines.append(f"**{lens}**: overall {overall:.1f}% — weakest variable is **{weak[0]}** ({weak[1]['pct']:.1f}%).")
        else:
            lines.append(f"**{lens}**: overall {overall:.1f}% — no variable breakdown (add vars on questions).")

    # Interplay prompt (what you described)
    # If goals low and relationship low, tie them
    lens_map = {x[0]: x for x in summary}
    if "goals" in trilens_results and "relationship" in trilens_results and "finance" in trilens_results:
        g = trilens_results["goals"]["overall"]
        r = trilens_results["relationship"]["overall"]
        f = trilens_results["finance"]["overall"]

        if r < 55 and f < 55:
            lines.append("Interplay: relationship stress + money stress often create a feedback loop. You don’t fix both at once—you stabilize one to stop the bleeding.")
        if g < 55 and r < 55:
            lines.append("Interplay: unclear goals + relationship strain usually means you’re negotiating the future every day. That burns both people out.")
        if g < 55 and f < 55:
            lines.append("Interplay: goals without a workable money plan turns into pressure. The plan doesn’t need to be big—it needs to be consistent.")

    lines.append("Recommendation: pick the **worst lens** first and run a 2-week micro-plan. Then re-test all three.")
    return lines


# =========================================================
# UI SCREENS
# =========================================================

def screen_home():
    st.title(APP_TITLE)
    st.caption("Choose a single lens run, or do the 3-lens clarity pass.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Single lens run")
        st.session_state.mode = "single"
        st.session_state.lens = st.selectbox("Lens", DEFAULT_LENSES, index=DEFAULT_LENSES.index(st.session_state.lens) if st.session_state.lens in DEFAULT_LENSES else 0)
        if st.button("Start (25 questions)", type="primary"):
            reset_single_run(keep_lens=True)
            qs = sample_questions(st.session_state.lens, k=25)
            st.session_state.active_questions = qs
            st.session_state.stage = "questions"
            st.rerun()

    with col2:
        st.subheader("3-lens clarity pass")
        st.session_state.mode = "trilens"
        st.write("Runs relationship → finance → goals (you can rename lenses).")
        if st.button("Start 3-lens pass", type="primary"):
            st.session_state.trilens_queue = DEFAULT_LENSES[:]  # order
            st.session_state.trilens_results = {}
            # start first lens
            st.session_state.lens = st.session_state.trilens_queue[0]
            reset_single_run(keep_lens=True)
            st.session_state.active_questions = sample_questions(st.session_state.lens, k=25)
            st.session_state.stage = "questions"
            st.rerun()

    st.divider()
    st.info("If you see blank screens: you probably haven’t added questions yet. Paste your question dicts into QUESTION_BANK.")


def screen_questions():
    lens = st.session_state.lens
    qs = st.session_state.active_questions

    st.header(f"Lens: {lens} — Main Questions")

    if not qs:
        st.warning("No questions found for this lens. Add questions to QUESTION_BANK.")
        if st.button("Back"):
            st.session_state.stage = "home"
            st.rerun()
        return

    idx = st.session_state.idx
    total = len(qs)
    q = qs[idx]

    st.progress((idx + 1) / total)
    st.write(f"**Q{idx+1}/{total}** — {q.get('var','General')}")
    st.write(q["text"])

    current = st.session_state.answers.get(q["id"], MIN_SCORE)
    val = st.slider("Answer", MIN_SCORE, MAX_SCORE, int(current), key=f"ans_{q['id']}")

    # save
    st.session_state.answers[q["id"]] = int(val)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Back", disabled=(idx == 0)):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    with col2:
        if st.button("Next", disabled=(idx >= total - 1)):
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()

    with col3:
        if st.button("Finish 25 & Pick follow-ups", type="primary"):
            # score base to choose followups
            overall, per_var, _rows, targets = score_run(lens, qs, st.session_state.answers)
            already = set([qq["id"] for qq in qs])
            fus = pick_followups(lens, targets, already_asked_ids=already, n=10)
            st.session_state.followup_questions = fus
            st.session_state.followup_answers = {}
            st.session_state.followup_idx = 0
            st.session_state.stage = "followups"
            st.rerun()


def screen_followups():
    lens = st.session_state.lens
    fqs = st.session_state.followup_questions

    st.header(f"Lens: {lens} — Follow-ups")

    if not fqs:
        st.warning("No follow-up questions were generated. Moving to export step.")
        st.session_state.stage = "export_form"
        st.rerun()
        return

    idx = st.session_state.followup_idx
    total = len(fqs)
    q = fqs[idx]

    st.progress((idx + 1) / total)
    st.write(f"**FU{idx+1}/{total}** — {q.get('var','General')}")
    st.write(q["text"])

    current = st.session_state.followup_answers.get(q["id"], MIN_SCORE)
    val = st.slider("Answer", MIN_SCORE, MAX_SCORE, int(current), key=f"fu_{q['id']}")

    st.session_state.followup_answers[q["id"]] = int(val)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Back", disabled=(idx == 0)):
            st.session_state.followup_idx = max(0, idx - 1)
            st.rerun()
    with col2:
        if st.button("Next", disabled=(idx >= total - 1)):
            st.session_state.followup_idx = min(total - 1, idx + 1)
            st.rerun()
    with col3:
        # THIS is the key: go to export_form BEFORE results2
        if st.button("Finish follow-ups & Re-score", type="primary", key="btn_fu_finish_v1"):
            st.session_state.stage = "export_form"
            st.rerun()


def screen_export_form():
    lens = st.session_state.lens

    base_qs = st.session_state.active_questions
    base_answers = st.session_state.answers

    fqs = st.session_state.followup_questions
    fu_answers = st.session_state.followup_answers

    merged_questions = base_qs[:] + fqs[:]
    merged_answers = dict(base_answers)
    merged_answers.update(fu_answers)

    overall2, per_var2, _rows2, targets2 = score_run(lens, merged_questions, merged_answers)

    # Store payload for results2 + tri-lens
    payload = {
        "lens": lens,
        "phase": "after_25_plus_10",
        "overall": float(overall2),
        "per_var": per_var2,
        "answers": merged_answers,
        "targets": targets2,
    }
    st.session_state.last_results = payload

    st.header("Save this run (Google Form)")
    st.caption("Fill the form, then hit Continue to Results.")

    if "PASTE_YOUR_EMBED_URL_HERE" in GOOGLE_FORM_EMBED_URL:
        st.warning("You must paste your Google Form EMBED url into GOOGLE_FORM_EMBED_URL at the top of the file.")
    else:
        st.markdown(
            f"""
            <iframe
            src="{GOOGLE_FORM_EMBED_URL}"
            width="100%"
            height="1200"
            frameborder="0"
            marginheight="0"
            marginwidth="0">
            Loading…
            </iframe>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.write("### Export (copy/paste)")
    st.code(payload, language="python")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Skip saving (continue anyway)"):
            st.session_state.stage = "results2"
            st.rerun()
    with col2:
        if st.button("Continue to Results", type="primary"):
            st.session_state.stage = "results2"
            st.rerun()


def screen_results2():
    payload = st.session_state.last_results
    lens = st.session_state.lens

    st.header(f"Results (after 25 + 10) — {lens}")

    if not payload:
        st.warning("No results payload found.")
        if st.button("Back"):
            st.session_state.stage = "home"
            st.rerun()
        return

    overall = payload["overall"]
    per_var = payload["per_var"]

    st.metric("Overall", f"{overall:.1f}%")

    if per_var:
        st.write("### Variables")
        for v, d in sorted(per_var.items(), key=lambda kv: kv[1]["pct"]):
            st.write(f"- **{v}**: {d['pct']:.1f}% (n={d.get('n',0)})")

    st.divider()
    st.write("### Interpretation")
    for line in interpret(per_var):
        st.write(f"- {line}")

    st.divider()

    colA, colB, colC = st.columns([2, 1, 1])

    with colA:
        if st.button("Run another 10 follow-ups", type="primary"):
            # choose targets from current per_var
            targets = payload.get("targets", [])
            already_ids = set([q["id"] for q in st.session_state.active_questions] + [q["id"] for q in st.session_state.followup_questions])
            next_fus = pick_followups(lens, targets, already_asked_ids=already_ids, n=10)
            st.session_state.followup_questions = next_fus
            st.session_state.followup_answers = {}
            st.session_state.followup_idx = 0
            st.session_state.stage = "followups"
            st.rerun()

    with colB:
        if st.button("New run (same lens)"):
            reset_single_run(keep_lens=True)
            st.session_state.active_questions = sample_questions(lens, k=25)
            st.session_state.stage = "questions"
            st.rerun()

    with colC:
        if st.button("Change lens"):
            reset_single_run(keep_lens=False)
            st.session_state.stage = "home"
            st.rerun()

    # If we're in tri-lens mode, store and proceed automatically to next lens or show tri-lens summary
    if st.session_state.mode == "trilens":
        # store
        st.session_state.trilens_results[lens] = payload

        # if there are more lenses in the queue, advance
        q = st.session_state.trilens_queue
        if lens in q:
            idx = q.index(lens)
            if idx < len(q) - 1:
                nxt = q[idx + 1]
                st.info(f"Next lens ready: {nxt}")
                if st.button("Continue to next lens", type="primary"):
                    st.session_state.lens = nxt
                    reset_single_run(keep_lens=True)
                    st.session_state.active_questions = sample_questions(nxt, k=25)
                    st.session_state.stage = "questions"
                    st.rerun()
            else:
                st.success("All lenses complete.")
                if st.button("View 3-lens clarity summary", type="primary"):
                    st.session_state.stage = "trilens_results"
                    st.rerun()


def screen_trilens_results():
    st.header("3-Lens Clarity Summary")

    results = st.session_state.trilens_results
    if not results:
        st.warning("No tri-lens results found.")
        if st.button("Back"):
            st.session_state.stage = "home"
            st.rerun()
        return

    st.write("### Cross-lens synthesis")
    for line in cross_lens_synthesis(results):
        st.write(f"- {line}")

    st.divider()
    st.write("### Raw exports")
    st.code(results, language="python")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start another 3-lens pass", type="primary"):
            st.session_state.trilens_queue = DEFAULT_LENSES[:]
            st.session_state.trilens_results = {}
            st.session_state.lens = st.session_state.trilens_queue[0]
            reset_single_run(keep_lens=True)
            st.session_state.active_questions = sample_questions(st.session_state.lens, k=25)
            st.session_state.stage = "questions"
            st.rerun()
    with col2:
        if st.button("Home"):
            st.session_state.stage = "home"
  
