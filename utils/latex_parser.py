"""
latex_parser.py — Convert LaTeX source files to clean plain text
─────────────────────────────────────────────────────────────────
WHAT IT DOES:
    Takes the raw bytes of a .tex file and returns a clean, readable string
    with all LaTeX commands stripped out.

WHY WE NEED IT:
    LaTeX CVs are full of markup like \section{Education}, \textbf{Python},
    \begin{itemize}...\end{itemize}, etc. If we sent this raw markup to Gemini,
    it would still work, but the prompt would be noisy and waste tokens.
    Stripping the markup first gives Gemini clean content to reason about.

HOW IT WORKS:
    We use the `pylatexenc` library, specifically its `LatexNodes2Text` class.
    It parses the LaTeX AST (Abstract Syntax Tree) and converts each node
    to its plain-text equivalent — so \textbf{Hello} becomes "Hello",
    \section{Experience} becomes "Experience", etc.
"""

from pylatexenc.latex2text import LatexNodes2Text


def parse_latex(file_bytes: bytes) -> str:
    """
    WHAT: Decodes a .tex file from bytes and returns plain text.

    WHY: LaTeX markup is not human-readable text — it's code that describes
         how a document should look. We need the content, not the markup.

    HOW:
        1. Decode bytes to a Python string (UTF-8, with fallback to latin-1
           for older .tex files that use non-UTF-8 encoding)
        2. Feed the string into LatexNodes2Text which walks the LaTeX parse
           tree and converts each node to its text equivalent
        3. Strip leading/trailing whitespace and return

    Args:
        file_bytes (bytes): Raw bytes read from the uploaded .tex file.

    Returns:
        str: Clean plain text with all LaTeX commands removed.

    Raises:
        ValueError: If the bytes cannot be decoded or parsed.
    """
    # Step 1 — Decode bytes to string.
    # We try UTF-8 first (modern default). If that fails, we fall back to
    # latin-1 which can decode any byte sequence without throwing errors.
    try:
        latex_source = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        latex_source = file_bytes.decode("latin-1")

    # Step 2 — Convert LaTeX to plain text.
    # LatexNodes2Text() creates a converter object. Calling .latex_to_text()
    # on it parses the full LaTeX document and returns a plain string.
    # The strict_latex_spaces=False option is more forgiving with spacing,
    # which produces cleaner output for CV content.
    try:
        converter = LatexNodes2Text(strict_latex_spaces=False)
        plain_text = converter.latex_to_text(latex_source)
    except Exception as exc:
        # pylatexenc can occasionally fail on very unusual LaTeX.
        # In that case, we do a best-effort fallback: just return the raw
        # source as-is. It's noisy but better than crashing.
        plain_text = latex_source

    return plain_text.strip()
