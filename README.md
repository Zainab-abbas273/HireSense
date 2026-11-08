# HireSense 📄

An AI-powered resume screener. Paste a job description, upload resumes (PDF or DOCX), and ResumeRank parses each resume with an LLM and ranks candidates by how well they match the job.

Built with Python, Streamlit, Groq and Pydantic.

<!-- Add a screenshot of your app here -->
<!-- ![ResumeRank screenshot](screenshot.png) -->

## Features

- Upload multiple PDF and DOCX resumes at once, or load them from a local `resumes/` folder
- Extracts structured data from the job description (department, responsibilities, skills, education)
- Parses each resume into validated JSON (name, email, skills, education, projects, certifications, experience)
- Scores every candidate from 0 to 100, with matching skills, missing skills, experience check and a short verdict
- Ranked results table with score bars, top 2 and lowest 2 candidates, and CSV export
- Per-resume error handling, so one bad file does not stop the whole batch
- Adjustable delay between API calls to avoid rate limits

## How it works

```
Job description ──► LLM ──► structured job (JSON)
                                   │
Resume (PDF/DOCX) ──► text ──► LLM ──► structured resume (JSON)
                                   │
                  structured job + structured resume ──► LLM ──► score + details
                                   │
                              ranked results
```

1. **Job extraction:** the job description is turned into JSON that follows a Pydantic schema.
2. **Resume parsing:** text is extracted from the file (`pypdf` / `python-docx`), then the LLM fills a resume schema based on meaning, not on exact section headings.
3. **Matching:** the LLM compares the structured resume with the structured job and returns a score and details.
4. **Ranking:** results are sorted by score and shown in the Streamlit app.

Pydantic validates every LLM response, so the app always works with clean, predictable data.

## Tech stack

| Part | Tool |
|------|------|
| Language | Python |
| Frontend | Streamlit |
| LLM API | Groq (`openai/gpt-oss-120b`) |
| Data validation | Pydantic |
| File reading | pypdf, python-docx |
| Data table | pandas |
| Package manager | uv |

## Project structure

```
proj1/
├── fe.py             # Streamlit app (main file)
├── .env              # your API key (not committed)
├── pyproject.toml    # project dependencies (managed by uv)
└── resumes/          # optional: resumes to screen
```

## Setup

### 1. Install uv

```
pip install uv
```

(Or follow the [official uv install guide](https://docs.astral.sh/uv/getting-started/installation/).)

### 2. Clone the repo and install dependencies

```
git clone https://github.com/<your-username>/ResumeRank.git
cd ResumeRank
uv sync
```

If you are starting from scratch instead:

```
uv init
uv add streamlit pandas groq python-dotenv pydantic pypdf python-docx
```

### 3. Add your Groq API key

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_api_key_here
```

You can get a free key from the [Groq console](https://console.groq.com/).

### 4. Run the app

```
uv run streamlit run fe.py
```

## Usage

1. Paste a job description in the sidebar.
2. Upload resumes, or keep the **resumes folder** option ticked to use files from `resumes/`.
3. Click **Analyze Resumes**.
4. View the ranking, open any candidate for details, and download the results as CSV.

## Limitations

- Scores come from an LLM, so they can change slightly between runs.
- Scanned PDFs (images without text) cannot be read, because the app does not use OCR.
- Each resume needs two LLM calls, so large batches are limited by API rate limits and speed.

## Future improvements

- Score each requirement separately with a fixed rubric for more consistent results
- OCR support for scanned resumes
- Upload the job description as a file
- Save past results and compare runs
- Skills comparison charts

## License

MIT
