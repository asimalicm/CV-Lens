"""
report_renderer.py — Render the Gemini evaluation results in Streamlit
───────────────────────────────────────────────────────────────────────
WHAT IT DOES:
    Takes the dict returned by gemini_client.analyze_cv() and renders it
    as a visually clear, well-organised report inside the Streamlit app.

WHY WE NEED IT:
    Separating the "display" logic from the "analysis" logic is a clean
    software pattern. gemini_client.py doesn't know or care about Streamlit,
    and app.py stays uncluttered. report_renderer.py owns everything
    visual about the results.

HOW IT WORKS:
    We use Streamlit's layout primitives:
    - st.columns()   → side-by-side layout
    - st.metric()    → the big number score card
    - st.progress()  → visual score bar
    - st.success/warning/error() → colour-coded verdict banner
    - st.expander()  → collapsible sections for detail
    - st.markdown()  → formatted text with bullet points
"""

import streamlit as st


# ─── Verdict colour mapping ───────────────────────────────────────────────────
# Maps each verdict string to the matching Streamlit alert function.
# st.success → green, st.warning → orange/yellow, st.error → red
VERDICT_DISPLAY = {
    "Strong Match":    ("success", "✅"),
    "Possible Match":  ("warning", "⚠️"),
    "Not a Match":     ("error",   "❌"),
}


def render_report(analysis: dict) -> None:
    """
    WHAT: Renders the full analysis report in the Streamlit UI.

    WHY: One function call from app.py renders everything — the score,
         the verdict, strengths, gaps, and the detailed breakdown.

    HOW:
        Layout is split into three visual zones:
        1. Hero row  — large score metric + verdict badge
        2. Detail row — two columns: Strengths (left) | Gaps (right)
        3. Expander  — per-criterion table for the detail-oriented reader

    Args:
        analysis (dict): The dict returned by gemini_client.analyze_cv().
            Expected keys: match_score, verdict, strengths, gaps, per_criterion
    """
    score         = analysis.get("match_score", 0)
    verdict       = analysis.get("verdict", "Not a Match")
    strengths     = analysis.get("strengths", [])
    gaps          = analysis.get("gaps", [])
    per_criterion = analysis.get("per_criterion", {})

    st.markdown("---")
    st.subheader("Analysis Report")

    # ── Zone 1: Score + Verdict ───────────────────────────────────────────────
    col_score, col_verdict = st.columns([1, 2])

    with col_score:
        # st.metric() renders a big bold number — perfect for a score display.
        # The delta shows whether the score is above/below the 50% midpoint.
        delta = score - 50
        delta_color = "normal" if delta >= 0 else "inverse"
        st.metric(
            label="Overall Match Score",
            value=f"{score} / 100",
            delta=f"{delta:+d} vs midpoint",
            delta_color=delta_color,
        )
        # Progress bar gives an instant visual sense of the score.
        # st.progress() takes a float 0.0–1.0.
        st.progress(score / 100)

    with col_verdict:
        # Look up the right colour and icon for this verdict.
        alert_type, icon = VERDICT_DISPLAY.get(verdict, ("error", "❓"))

        # Call the correct st.success/st.warning/st.error function dynamically.
        # getattr(st, "success") is the same as calling st.success() directly.
        getattr(st, alert_type)(f"{icon} **Verdict: {verdict}**")

        # Add a plain-language interpretation of the score band
        if score >= 70:
            st.markdown(
                "This candidate is a **strong fit** for the role. "
                "Recommend proceeding to interview."
            )
        elif score >= 40:
            st.markdown(
                "This candidate has **partial fit**. "
                "Consider a screening call to assess gaps."
            )
        else:
            st.markdown(
                "This candidate **does not meet** the core requirements. "
                "Significant skill gaps identified."
            )

    st.markdown("---")

    # ── Zone 2: Strengths and Gaps side by side ───────────────────────────────
    col_strengths, col_gaps = st.columns(2)

    with col_strengths:
        st.markdown("### ✅ Strengths")
        if strengths:
            for strength in strengths:
                st.markdown(f"- {strength}")
        else:
            st.markdown("*No significant strengths identified.*")

    with col_gaps:
        st.markdown("### ⚠️ Gaps")
        if gaps:
            for gap in gaps:
                st.markdown(f"- {gap}")
        else:
            st.markdown("*No significant gaps identified.*")

    st.markdown("---")

    # ── Zone 3: Per-criterion breakdown ──────────────────────────────────────
    # This is the most detailed section. We put it in an expander so it
    # doesn't overwhelm users who just want the top-level result.
    with st.expander("📋 Detailed Criterion-by-Criterion Breakdown", expanded=False):
        if per_criterion:
            for criterion, assessment in per_criterion.items():
                # Use a simple emoji to visually categorise each assessment.
                # If the assessment mentions "not" or "missing", flag it red.
                lower = assessment.lower()
                if any(word in lower for word in ["not mentioned", "missing", "no evidence", "not found", "absent"]):
                    icon = "🔴"
                elif any(word in lower for word in ["partial", "limited", "some", "basic"]):
                    icon = "🟡"
                else:
                    icon = "🟢"

                st.markdown(f"{icon} **{criterion}**")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{assessment}")
                st.markdown("")  # blank line for spacing
        else:
            st.markdown("*No per-criterion data available.*")
