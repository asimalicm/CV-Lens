"""
file_router.py — Central dispatcher for CV file uploads
────────────────────────────────────────────────────────
WHAT IT DOES:
    Looks at the file extension of whatever the user uploaded and sends it
    to the right extraction pipeline. Think of it like a post office sorting
    room — every package (file) gets sent to the correct department.

WHY WE NEED IT:
    We support three very different file formats (.tex, .pdf, images), and each
    one needs a completely different tool to extract the CV content. Rather than
    stuffing all that logic into app.py, we centralise the "what do I do with
    this file?" question here in one place.

HOW IT WORKS:
    1. We look up the file extension in SUPPORTED_EXTENSIONS dict.
    2. Depending on the category ("latex", "pdf", "image"), we call the
       matching parser and return a tuple describing what we extracted.
    3. The caller (app.py) doesn't need to know which parser was used —
       it just passes the result to gemini_client.py.

RETURN VALUE:
    For text-based formats (.tex, .pdf):
        ("text", <plain text string>)

    For image formats (.png, .jpg, .jpeg, .webp):
        ("image", <raw bytes>, <mime type string>)
        e.g. ("image", b"...", "image/jpeg")
"""

from utils.latex_parser import parse_latex
from utils.pdf_parser import parse_pdf

# ─── Supported file extensions ───────────────────────────────────────────────
# Maps each file extension (lowercase, no dot) to a category string.
# Adding support for a new format is as simple as adding a line here.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    "tex":  "latex",
    "pdf":  "pdf",
    "png":  "image",
    "jpg":  "image",
    "jpeg": "image",
    "webp": "image",
}

# Maps image extensions to the correct MIME type string.
# Gemini's vision API needs the MIME type to know how to decode the image bytes.
IMAGE_MIME_TYPES: dict[str, str] = {
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def route_file(filename: str, file_bytes: bytes) -> tuple:
    """
    WHAT: Detects the file type and returns extracted content in a standard format.

    WHY: Centralises the format-detection logic so app.py stays clean.

    HOW:
        - Extract the extension from the filename (e.g. "cv.pdf" → "pdf")
        - Look it up in SUPPORTED_EXTENSIONS
        - Call the appropriate parser or just return the raw bytes for images
        - Return a uniform tuple that gemini_client.py knows how to handle

    Args:
        filename   (str):   The original filename from the upload widget.
        file_bytes (bytes): The raw bytes of the uploaded file.

    Returns:
        tuple: ("text", str) for .tex/.pdf
               ("image", bytes, str) for image files

    Raises:
        ValueError: If the extension is not in SUPPORTED_EXTENSIONS.
    """
    # Pull the extension off the end of the filename, lowercase it for safety.
    # "CV_Final_v2.TEX" → "tex"
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(f".{e}" for e in SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported formats are: {supported}"
        )

    category = SUPPORTED_EXTENSIONS[ext]

    # ── LaTeX path ────────────────────────────────────────────────────────────
    # pylatexenc understands LaTeX markup and strips \section{}, \textbf{}, etc.
    if category == "latex":
        text = parse_latex(file_bytes)
        return ("text", text)

    # ── PDF path ──────────────────────────────────────────────────────────────
    # pymupdf reads the actual text layer stored inside the PDF binary.
    if category == "pdf":
        text = parse_pdf(file_bytes)
        return ("text", text)

    # ── Image path ────────────────────────────────────────────────────────────
    # We do NOT extract text from images ourselves.
    # Instead we hand the raw bytes directly to Gemini Vision, which is far
    # more accurate than any OCR tool for understanding CV layout and content.
    if category == "image":
        mime_type = IMAGE_MIME_TYPES[ext]
        return ("image", file_bytes, mime_type)
