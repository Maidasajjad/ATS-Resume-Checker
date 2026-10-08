# 📄 ATS Resume Checker

A Streamlit app that analyzes a resume with Google Gemini and returns:

- An **estimated ATS score** (0–100) with a breakdown (keywords, impact, formatting, structure, readability)
- **Missing keywords** (optionally matched against a pasted job description)
- **Prioritized improvements** with before/after rewrites
- A downloadable Markdown report

> The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.

## Features
- Upload PDF, DOCX or TXT (max 5 MB)
- Optional job description for tailored keyword matching
- Rule-based checks (email, phone, sections, bullets, quantified results) combined with AI analysis
- Detects scanned/image-only PDFs that ATS systems cannot read
- Overall score is computed in code from category scores, so it stays consistent

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then provide it **one** of these ways:

1. Paste it into the sidebar when the app runs, or
2. Environment variable:
   - macOS/Linux: `export GEMINI_API_KEY="your_key"`
   - Windows PowerShell: `$env:GEMINI_API_KEY="your_key"`
3. Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```

Start the app:

```bash
streamlit run app.py
```

## Configuration
| Setting | How | Default |
|---|---|---|
| API key | sidebar / `GEMINI_API_KEY` env var / Streamlit secret | none |
| Model | sidebar field or `GEMINI_MODEL` env var | `gemini-3.5-flash` (falls back to `gemini-2.5-flash` if unavailable) |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public, or private with access granted).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

**Never commit your API key.** Keep `.streamlit/secrets.toml` and `.env` in `.gitignore`.

## Project structure
```
app.py             # Streamlit UI + analysis logic
requirements.txt   # Python dependencies
README.md
```

## Privacy
Resume text is sent to the Gemini API for analysis. The app does not store uploaded files.
