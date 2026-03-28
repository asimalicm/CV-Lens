"""
gemini_client.py — Send CV data to Gemini AI and get a structured evaluation
─────────────────────────────────────────────────────────────────────────────
WHAT IT DOES:
    Takes the output of file_router.py (either plain text or image bytes) and
    the job criteria dict, builds a carefully engineered prompt, sends it to
    Google's Gemini 2.0 Flash model, and returns a structured Python dict with
    the evaluation results.

WHY WE NEED IT:
    The Gemini API can be called in two fundamentally different ways:
    - Text mode: send a text string, get a text response
    - Vision mode: send an image + text together, Gemini "sees" the image

    This module handles both modes and always returns the same dict shape,
    so the rest of the app doesn't need to care which mode was used.

HOW IT WORKS:
    1. Configure the Gemini SDK with the API key
    2. Create a GenerativeModel with JSON output mode enabled
    3. Build the appropriate prompt depending on whether we have text or image
    4. Call Gemini and parse the JSON response
    5. Return a Python dict with: match_score, verdict, strengths, gaps, per_criterion

JSON OUTPUT SCHEMA (what Gemini is asked to return):
    {
        "match_score": 0-100 integer,
        "verdict": "Strong Match" | "Possible Match" | "Not a Match",
        "strengths": ["list of specific strength strings"],
        "gaps": ["list of specific gap strings"],
        "per_criterion": {
            "Python": "5 years of experience evidenced in projects section",
            "Docker": "Not mentioned anywhere in the CV",
            ...
        }
    }
"""

import json
import re
import io

# google-genai is the new official Python SDK (replaces google-generativeai).
# The key difference: instead of global genai.configure() + genai.GenerativeModel(),
# you create a Client instance per call and use client.models.generate_content().
from google import genai
from google.genai import types
import yaml
from PIL import Image


# ─── The JSON schema we tell Gemini to follow ────────────────────────────────
# Providing the schema explicitly in the prompt dramatically improves
# how consistently Gemini formats its response.
JSON_SCHEMA = """
{
    "match_score": <integer between 0 and 100>,
    "verdict": <"Strong Match" if score >= 70, "Possible Match" if 40-69, "Not a Match" if below 40>,
    "strengths": [<list of specific, evidence-based strength strings>],
    "gaps": [<list of specific missing skills or experience strings>],
    "per_criterion": {
        "<each required skill or responsibility>": "<brief assessment string>"
    }
}
"""

# ─── Evaluation instructions shared by both text and vision prompts ───────────
EVALUATION_INSTRUCTIONS = """
You are an expert hiring manager and technical recruiter with 15 years of experience.
Your job is to evaluate a candidate's CV against the provided job criteria.

Rules for your evaluation:
- Be specific: reference actual content from the CV, not generic statements
- Be fair: give credit for transferable skills and equivalent experience
- Be honest: clearly flag missing requirements as gaps
- match_score should reflect overall fit (100 = perfect match, 0 = completely wrong profile)
- per_criterion must have one entry for EVERY required skill and EVERY key responsibility
- Return ONLY valid JSON. No explanation text, no markdown code fences, no extra keys.

The JSON must EXACTLY match this schema:
{schema}
""".strip()


def _build_text_prompt(cv_text: str, criteria_yaml: str) -> str:
    """
    WHAT: Builds the full prompt string for text-mode evaluation (.tex / .pdf).

    WHY: The prompt is the most important part of working with LLMs. A well-
         structured prompt with clear sections and explicit instructions produces
         far better results than dumping everything in one paragraph.

    HOW:
        We use clear section headers (--- JOB CRITERIA ---, --- CV TEXT ---)
        as "anchors" so Gemini can easily find each part of the input.

        If the CV text starts with our [NOTE: raw LaTeX] prefix (written by
        latex_parser.py's Layer 3 fallback), we add an extra reminder that
        Gemini should parse through the LaTeX markup to find the candidate's
        actual content. This prevents Gemini from treating the markup itself
        as the candidate's experience.
    """
    instructions = EVALUATION_INSTRUCTIONS.format(schema=JSON_SCHEMA)

    # Detect when latex_parser.py fell back to raw LaTeX source.
    # In that case, add an extra explicit instruction so Gemini doesn't
    # mistake LaTeX command names for the candidate's skills/experience.
    latex_note = ""
    if cv_text.startswith("[NOTE: The following CV is provided as raw LaTeX"):
        latex_note = (
            "\nIMPORTANT: The CV below is raw LaTeX source code. "
            "Read through the LaTeX markup (\\section, \\textbf, \\item, etc.) "
            "and evaluate the candidate's actual experience and skills as they "
            "would appear in a rendered CV. Do not treat LaTeX command names as "
            "the candidate's content.\n"
        )

    return f"""{instructions}{latex_note}

--- JOB CRITERIA ---
{criteria_yaml}

--- CV TEXT ---
{cv_text}
"""


def _build_vision_prompt(criteria_yaml: str) -> str:
    """
    WHAT: Builds the text portion of the vision-mode prompt (for image CVs).

    WHY: In vision mode, Gemini receives both the image AND a text prompt.
         The image is passed separately as inline data, so this function only
         constructs the text part of the prompt.

    HOW:
        We add explicit instruction to "read the CV in the image" because
        Gemini needs to be told that the image IS the CV, not just background.
    """
    instructions = EVALUATION_INSTRUCTIONS.format(schema=JSON_SCHEMA)
    return f"""{instructions}

The image provided contains a candidate's CV. Read all content in the image carefully.

--- JOB CRITERIA ---
{criteria_yaml}

Now evaluate the CV shown in the image against the criteria above.
"""


def _parse_json_response(response_text: str) -> dict:
    """
    WHAT: Parses the JSON string returned by Gemini into a Python dict.

    WHY: Even with response_mime_type="application/json", Gemini occasionally
         wraps the JSON in a markdown code block (```json ... ```). This
         function handles both cases gracefully.

    HOW:
        1. Try to parse the text directly as JSON
        2. If that fails, use regex to find a JSON block inside markdown fences
        3. If both fail, raise a clear error

    Args:
        response_text (str): The raw string from Gemini's response.

    Returns:
        dict: The parsed evaluation results.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # Attempt 1: direct parse (the happy path when JSON mode works cleanly)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract JSON from inside a markdown code fence
    # Pattern: ```json\n{ ... }\n``` or ``` { ... } ```
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Both attempts failed — the response is malformed
    raise ValueError(
        f"Gemini returned a response that could not be parsed as JSON.\n"
        f"Raw response (first 300 chars): {response_text[:300]}"
    )


def analyze_cv(routed_data: tuple, job_criteria: dict, api_key: str) -> dict:
    """
    WHAT: The main public function — sends the CV to Gemini and returns
          the structured evaluation dict.

    WHY: This is the bridge between the file parsing layer and the result
         display layer. Everything else in the app just calls this one function.

    HOW:
        1. Configure the Gemini SDK with the provided API key
        2. Create a model with JSON output mode forced on
        3. Convert job_criteria dict to a YAML string (readable by humans AND AI)
        4. Detect whether we have text or image data (from the tuple's first element)
        5. Build the right prompt, call Gemini, parse and return the result

    Args:
        routed_data (tuple):
            ("text", str) for .tex and .pdf files
            ("image", bytes, str) for image files
        job_criteria (dict): The loaded job_criteria.yaml content.
        api_key (str): Your Gemini API key.

    Returns:
        dict: {
            "match_score": int,
            "verdict": str,
            "strengths": list[str],
            "gaps": list[str],
            "per_criterion": dict[str, str]
        }

    Raises:
        ValueError: If the API key is missing or the response can't be parsed.
        google.api_core.exceptions.GoogleAPIError: On API call failures.
    """
    if not api_key:
        raise ValueError(
            "No Gemini API key found. "
            "Set GEMINI_API_KEY in your .env file (local) or Streamlit secrets (cloud)."
        )

    # Step 1 — Create a Client instance with our API key.
    # In the new google-genai SDK there is no global configure() call.
    # Instead you create a Client object and every call flows through it.
    # This is safer (no hidden global state) and makes testing easier.
    client = genai.Client(api_key=api_key)

    # Step 2 — Define the generation config using types.GenerateContentConfig.
    # response_mime_type="application/json" forces Gemini to return pure JSON.
    # temperature=0.2 keeps answers consistent and factual (less creative randomness).
    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )

    # Step 3 — Serialise the job criteria to YAML for the prompt.
    # YAML is much more readable than JSON for nested data, and LLMs handle
    # it very well because it looks like structured English.
    criteria_yaml = yaml.dump(job_criteria, default_flow_style=False, allow_unicode=True)

    # Step 4 — Detect input type and call Gemini with the right approach.
    input_type = routed_data[0]

    if input_type == "text":
        # ── Text mode (.tex or .pdf) ──────────────────────────────────────────
        _, cv_text = routed_data
        prompt = _build_text_prompt(cv_text, criteria_yaml)

        # client.models.generate_content() replaces model.generate_content().
        # The model name is now passed directly here instead of at model creation.
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=generation_config,
        )

    elif input_type == "image":
        # ── Vision mode (image files) ─────────────────────────────────────────
        _, image_bytes, mime_type = routed_data

        # In the new SDK, inline image data is passed as a types.Part.
        # types.Part.from_bytes() wraps raw bytes + mime_type into the correct
        # structure without needing PIL as an intermediary for the API call.
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        prompt_text = _build_vision_prompt(criteria_yaml)

        # Pass a list of [text_string, image_part] as the multimodal contents.
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt_text, image_part],
            config=generation_config,
        )

    else:
        raise ValueError(f"Unknown input type from file_router: '{input_type}'")

    # Step 5 — Parse and return the response.
    return _parse_json_response(response.text)
