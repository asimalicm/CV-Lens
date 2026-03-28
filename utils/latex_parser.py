"""
latex_parser.py — Convert LaTeX source files to clean plain text
─────────────────────────────────────────────────────────────────
WHAT IT DOES:
    Takes the raw bytes of a .tex file and returns the best possible
    plain-text representation of the CV content.

WHY WE NEED IT:
    LaTeX CVs are full of markup like \\section{Education}, \\textbf{Python},
    \\begin{itemize}...\\end{itemize}, etc. Gemini can read raw LaTeX, but
    giving it clean plain text produces more consistent evaluations.

HOW IT WORKS (three-layer fallback strategy):
    Layer 1 — pylatexenc (AST-based):
        The most accurate converter. Parses the full LaTeX parse tree and
        converts each node to its text equivalent. Fails on complex or
        non-standard CV templates.

    Layer 2 — regex stripping (pattern-based):
        If pylatexenc produces suspiciously little content, we apply a
        sequence of regex substitutions that strip LaTeX commands while
        preserving the text inside curly-brace arguments.
        Less elegant than AST parsing, but very reliable.

    Layer 3 — raw LaTeX source:
        If even the regex approach yields too little text, we return the
        original LaTeX source as-is. Gemini understands LaTeX natively, so
        this is always a valid last resort. We prepend a note so the AI
        knows to treat the content as LaTeX markup.
"""

import re

from pylatexenc.latex2text import LatexNodes2Text


# ─── Minimum viable content threshold ────────────────────────────────────────
# If the extracted text is shorter than this, we assume the parser failed
# and move to the next fallback layer.
_MIN_CONTENT_CHARS = 150


def _regex_strip_latex(latex_source: str) -> str:
    """
    WHAT: A fast, regex-based LaTeX-to-text converter used as a fallback.

    WHY: pylatexenc can silently fail on non-standard LaTeX commands that
         are common in CV templates (e.g. moderncv, awesomecv). Regex
         stripping is dumb but reliable — it just removes command syntax
         and keeps the human-readable text.

    HOW (step by step):
        1. Strip LaTeX comments (% to end of line)
        2. Remove \\begin{...} and \\end{...} environment markers
        3. For commands with arguments like \\textbf{Python}, keep the inner
           text: Python
        4. Remove bare commands like \\newpage, \\hline, etc.
        5. Remove leftover curly braces
        6. Collapse excess blank lines

    Args:
        latex_source (str): Raw LaTeX source as a string.

    Returns:
        str: Approximate plain text with markup stripped.
    """
    # Step 1 — Remove comments (% to end of line)
    text = re.sub(r"%.*$", "", latex_source, flags=re.MULTILINE)

    # Step 2 — Remove \begin{...} / \end{...} environment wrappers.
    # These mark structural blocks like itemize, document, etc. The content
    # inside is preserved by the steps below.
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)

    # Step 3 — Unwrap single-argument commands: \cmd{content} → content.
    # We repeat this several times to handle deeply nested commands like
    # \textbf{\textit{Senior Python Developer}}
    for _ in range(5):
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)

    # Step 4 — Remove bare commands that take no arguments (e.g. \newpage)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

    # Step 5 — Remove leftover curly braces and special LaTeX characters
    text = re.sub(r"[{}\\]", " ", text)

    # Step 6 — Normalise whitespace: collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def parse_latex(file_bytes: bytes) -> str:
    """
    WHAT: Decodes a .tex file from bytes and returns the best plain text
          representation, using a three-layer fallback strategy.

    WHY: No single LaTeX parser handles every CV template correctly. The
         three-layer approach means we always return something useful rather
         than silently returning empty content (which was the original bug —
         it caused Gemini to see an empty CV and score it 0/100).

    HOW:
        Layer 1: pylatexenc AST parser
        Layer 2: regex stripping (if Layer 1 yields < _MIN_CONTENT_CHARS)
        Layer 3: raw LaTeX source with a note (if Layer 2 also yields too little)

    Args:
        file_bytes (bytes): Raw bytes read from the uploaded .tex file.

    Returns:
        str: The best plain text we could extract. Never empty — at minimum
             returns the raw LaTeX source.

    Raises:
        ValueError: If the bytes cannot be decoded at all.
    """
    # ── Decode bytes to string ────────────────────────────────────────────────
    # UTF-8 is the modern standard. latin-1 is a safe fallback that can decode
    # any byte sequence without raising an error (every byte maps to a char).
    try:
        latex_source = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        latex_source = file_bytes.decode("latin-1")

    # ── Layer 1: pylatexenc AST-based conversion ──────────────────────────────
    plain_text = ""
    try:
        converter = LatexNodes2Text(strict_latex_spaces=False)
        plain_text = converter.latex_to_text(latex_source).strip()
    except Exception:
        # pylatexenc raised an error — move straight to Layer 2
        plain_text = ""

    if len(plain_text) >= _MIN_CONTENT_CHARS:
        return plain_text

    # ── Layer 2: regex-based stripping ───────────────────────────────────────
    # pylatexenc gave us too little — try the simpler regex approach.
    regex_text = _regex_strip_latex(latex_source)

    if len(regex_text) >= _MIN_CONTENT_CHARS:
        return regex_text

    # ── Layer 3: raw LaTeX source ─────────────────────────────────────────────
    # Both parsers failed to produce enough content. Return the raw source with
    # a clear note so Gemini knows it is reading LaTeX markup, not plain text.
    # Gemini is trained on LaTeX and handles it well.
    note = (
        "[NOTE: The following CV is provided as raw LaTeX source. "
        "Please read through the LaTeX markup and evaluate the candidate's "
        "experience, skills, and background as if reading their CV normally.]\n\n"
    )
    return note + latex_source
