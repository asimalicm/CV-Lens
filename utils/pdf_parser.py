"""
pdf_parser.py — Extract plain text from PDF files using pymupdf
────────────────────────────────────────────────────────────────
WHAT IT DOES:
    Takes the raw bytes of a PDF file and returns all the text content
    as a single clean string.

WHY WE NEED IT:
    PDFs are not plain text files — they store content as positioned drawing
    commands (think of it like a canvas where each character is placed at
    exact X,Y coordinates). To get the actual words out, we need a library
    that understands the PDF binary format.

    We use `pymupdf` (imported as `fitz`) because:
    - It is one of the fastest and most accurate PDF text extractors
    - It handles multi-column layouts (common in CVs) better than simpler tools
    - It works on PDF bytes directly, no need to save a temp file to disk

HOW IT WORKS:
    1. Open the PDF from the raw bytes using fitz.open(stream=...)
    2. Loop over every page and extract text from each
    3. Join all pages together with a separator and return
"""

import fitz  # fitz is the import name for the pymupdf package


def parse_pdf(file_bytes: bytes) -> str:
    """
    WHAT: Reads a PDF from bytes and returns all its text content.

    WHY: We need plain text to build the prompt for Gemini. The PDF binary
         format is unreadable without a parser.

    HOW:
        - fitz.open() accepts raw bytes via the `stream` parameter
        - We iterate over each page object and call .get_text()
        - "text" mode preserves reading order (top-to-bottom, left-to-right)
        - Pages are joined with a clear separator so Gemini can tell where
          one page ends and the next begins

    Args:
        file_bytes (bytes): Raw bytes read from the uploaded PDF file.

    Returns:
        str: All text content extracted from the PDF, pages joined by a
             newline separator.

    Raises:
        ValueError: If the PDF cannot be opened or has no extractable text.
    """
    # Open the PDF directly from memory (no temp file needed).
    # filetype="pdf" tells fitz what format to expect.
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

    pages_text = []

    # Loop over each page. fitz uses 0-based page indexing.
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)

        # get_text("text") returns the text in reading order.
        # Other modes like "blocks" or "words" give more structure,
        # but plain "text" is cleanest for feeding into an LLM prompt.
        page_text = page.get_text("text")

        if page_text.strip():  # skip completely blank pages
            pages_text.append(page_text.strip())

    pdf_document.close()

    if not pages_text:
        # This happens with scanned PDFs (images embedded in PDF).
        # We raise a clear error so the UI can tell the user to upload
        # the image directly instead.
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document. "
            "Please upload the CV as an image (.png or .jpg) instead."
        )

    # Join pages with a clear visual separator
    return "\n\n--- Page Break ---\n\n".join(pages_text)
