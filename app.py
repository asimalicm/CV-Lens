"""
app.py — CV Analyzer AI: Main Streamlit Application Entry Point
───────────────────────────────────────────────────────────────
WHAT IT DOES:
    This is the top-level file that Streamlit runs. It wires together all the
    utility modules and renders the complete web UI from top to bottom.

WHY THIS FILE EXISTS:
    Streamlit works by running a Python script from top to bottom every time
    the user interacts with the page. This file is the "conductor" — it calls
    the right utility functions in the right order and handles errors gracefully.

HOW THE FLOW WORKS:
    1. Load configuration (API key + job criteria YAML)
    2. Show the sidebar with instructions and job criteria preview
    3. Show the main upload widget
    4. When a file is uploaded:
       a. file_router.py detects the format and extracts the content
       b. gemini_client.py sends it to Gemini and returns structured results
       c. report_renderer.py renders the results in the UI
    5. Results are cached in st.session_state so they survive Streamlit reruns
       (Streamlit re-runs the script from top to bottom on every UI interaction)

HOW TO RUN LOCALLY:
    streamlit run app.py
"""

import os
import yaml
import streamlit as st
from dotenv import load_dotenv

from utils.file_router import route_file
from utils.gemini_client import analyze_cv
from utils.report_renderer import render_report

# ─── Page configuration ───────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script — Streamlit enforces this.
st.set_page_config(
    page_title="CV Analyzer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Load API key ─────────────────────────────────────────────────────────────
# We support two sources for the API key so the app works both locally
# and on Streamlit Community Cloud:
#   - Local development: reads from .env file via python-dotenv
#   - Streamlit Cloud:   reads from the app's Secrets settings (st.secrets)
#
# The try/except around st.secrets is important: if there is no secrets.toml
# file (i.e. running locally), Streamlit raises a FileNotFoundError rather
# than returning None, so we must catch it.
load_dotenv()  # Load .env file if it exists (has no effect if it doesn't)

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        api_key = ""


# ─── Load job criteria ────────────────────────────────────────────────────────
# We load this once at startup. If the file is missing or malformed,
# we show a clear error instead of an unhelpful Python traceback.
@st.cache_data  # Cache so the file isn't re-read on every Streamlit rerun
def load_job_criteria() -> dict:
    """
    WHAT: Reads and parses config/job_criteria.yaml.

    WHY: Job criteria is static configuration — it only changes when you
         deliberately edit the YAML file, not on every page interaction.
         Caching it avoids unnecessary disk reads.

    Returns:
        dict: The parsed YAML content as a Python dictionary.
    """
    with open("config/job_criteria.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

try:
    job_criteria = load_job_criteria()
except FileNotFoundError:
    st.error(
        "Could not find `config/job_criteria.yaml`. "
        "Make sure you're running this app from the project root directory."
    )
    st.stop()  # Halt execution — the rest of the app can't work without criteria


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 CV Analyzer AI")
    st.markdown("Evaluate any CV against a job description using **Gemini AI**.")
    st.markdown("---")

    # Show what role we are currently hiring for
    st.markdown("### Current Role")
    st.info(f"**{job_criteria.get('job_title', 'Not set')}**")

    # Show required skills as a readable list
    st.markdown("**Required Skills:**")
    for skill in job_criteria.get("required_skills", []):
        st.markdown(f"- {skill}")

    st.markdown("**Preferred Skills:**")
    for skill in job_criteria.get("preferred_skills", []):
        st.markdown(f"- {skill}")

    exp = job_criteria.get("minimum_experience_years", "N/A")
    st.markdown(f"**Min. Experience:** {exp} years")

    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown(
        "1. Upload a CV (`.tex`, `.pdf`, or image)\n"
        "2. Wait for the AI analysis (~10–20 seconds)\n"
        "3. Review the match score and report\n\n"
        "To change the job criteria, edit `config/job_criteria.yaml`."
    )

    st.markdown("---")

    # API key status indicator
    if api_key:
        st.success("✅ Gemini API key loaded")
    else:
        st.error(
            "❌ No Gemini API key found.\n\n"
            "Add `GEMINI_API_KEY=...` to your `.env` file (local) "
            "or Streamlit Secrets (cloud)."
        )


# ─── Main content area ────────────────────────────────────────────────────────
st.title("CV Analyzer AI")
st.markdown(
    "Upload a candidate's CV and get an AI-powered evaluation against the "
    f"**{job_criteria.get('job_title', 'defined role')}** job requirements."
)

# Show a warning banner if there's no API key — better than a confusing error later
if not api_key:
    st.warning(
        "⚠️ No Gemini API key configured. "
        "The app will not be able to analyse CVs until a key is provided. "
        "See the sidebar for instructions."
    )

st.markdown("---")

# ─── File uploader ────────────────────────────────────────────────────────────
# We accept all supported formats in one widget.
# type= filters what the OS file picker shows (but doesn't validate server-side,
# which is why file_router.py also validates the extension).
uploaded_file = st.file_uploader(
    label="Upload CV file",
    type=["tex", "pdf", "png", "jpg", "jpeg", "webp"],
    help="Supported formats: LaTeX (.tex), PDF (.pdf), or images (.png, .jpg, .jpeg, .webp)",
)

# ─── Analysis trigger ─────────────────────────────────────────────────────────
# We use st.session_state to store results so they persist across reruns.
# Without this, every time the user scrolls or clicks anything, Streamlit
# re-runs the script and the results would disappear.
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

if uploaded_file is not None:
    # Detect if this is a new file (different from what we last analysed).
    # If the same file is re-uploaded we don't re-analyse (saves API calls).
    is_new_file = uploaded_file.name != st.session_state.last_filename

    if is_new_file:
        # Clear previous results when a new file comes in
        st.session_state.analysis_result = None

    # Show file info
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        file_ext = uploaded_file.name.rsplit(".", 1)[-1].upper()
        st.markdown(f"**File:** `{uploaded_file.name}` &nbsp;|&nbsp; **Format:** `{file_ext}`")

    with col_btn:
        analyse_clicked = st.button(
            "🔍 Analyse CV",
            type="primary",
            disabled=not api_key,
            use_container_width=True,
        )

    # ── Run the analysis pipeline ─────────────────────────────────────────────
    if analyse_clicked:
        if not api_key:
            st.error("Cannot analyse: no Gemini API key configured.")
        else:
            file_bytes = uploaded_file.read()

            # We wrap everything in a try/except so users see friendly
            # error messages instead of raw Python tracebacks.
            try:
                with st.spinner("Extracting CV content..."):
                    # Step 1 — Detect format and extract content
                    routed_data = route_file(uploaded_file.name, file_bytes)

                with st.spinner("Gemini is reading the CV and evaluating the candidate... (~10–20s)"):
                    # Step 2 — Send to Gemini and get structured results
                    result = analyze_cv(routed_data, job_criteria, api_key)

                # Step 3 — Store results in session state so they survive reruns
                st.session_state.analysis_result = result
                st.session_state.last_filename = uploaded_file.name

            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(
                    f"An unexpected error occurred: {e}\n\n"
                    "Please check your API key and try again."
                )

    # ── Display results (if we have them) ─────────────────────────────────────
    if st.session_state.analysis_result is not None:
        render_report(st.session_state.analysis_result)

else:
    # ── Empty state — shown when no file has been uploaded yet ────────────────
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0; color: #888;">
            <h3>👆 Upload a CV to get started</h3>
            <p>Drag and drop a file above, or click <em>Browse files</em>.</p>
            <p>Supported: <code>.tex</code> &nbsp; <code>.pdf</code> &nbsp;
               <code>.png</code> &nbsp; <code>.jpg</code> &nbsp; <code>.webp</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
