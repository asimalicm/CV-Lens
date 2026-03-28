# cv-lens

A Streamlit web app that evaluates a candidate's CV against a job description using the **Gemini 2.0 Flash** AI model. Supports LaTeX (`.tex`), PDF, and image (`.png`, `.jpg`, `.webp`) CVs.

## How It Works

1. Upload a CV file (`.tex`, `.pdf`, or image)
2. The tool auto-detects the format and extracts content using the best method
3. Gemini AI evaluates the CV against the job criteria in `config/job_criteria.yaml`
4. A detailed report is shown: match score, verdict, strengths, gaps, and per-criterion breakdown

| Format | Extraction Method |
|--------|------------------|
| `.tex` | `pylatexenc` — understands LaTeX commands |
| `.pdf` | `pymupdf` — accurate text extraction |
| Image  | Gemini Vision — reads the CV visually (no OCR needed) |

## Local Setup

**1. Clone the repo and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Create your `.env` file:**
```bash
cp .env.example .env
# Edit .env and paste your Gemini API key
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

**3. Edit the job criteria** in `config/job_criteria.yaml` to match the role you are hiring for.

**4. Run the app:**
```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub
3. Click **New app** → select your repo → set `app.py` as the entry point
4. Go to **Settings → Secrets** and add:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
5. Deploy — done!

## Customising the Job Criteria

Edit `config/job_criteria.yaml`. No Python knowledge needed:

```yaml
job_title: "Your Job Title"
required_skills: ["Skill A", "Skill B"]
preferred_skills: ["Nice to have"]
minimum_experience_years: 3
education_requirement: "Degree or equivalent"
key_responsibilities:
  - "Responsibility 1"
  - "Responsibility 2"
```

## Project Structure

```
cv-lens/
├── app.py                     # Streamlit UI entry point
├── config/
│   └── job_criteria.yaml      # Job requirements (edit this)
├── utils/
│   ├── file_router.py         # Detects file type, calls right extractor
│   ├── latex_parser.py        # .tex → plain text
│   ├── pdf_parser.py          # .pdf → plain text
│   ├── gemini_client.py       # Gemini API calls (text + vision)
│   └── report_renderer.py     # Streamlit result display
├── requirements.txt
├── .env.example
└── README.md
```
