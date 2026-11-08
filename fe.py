import os
import json
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document

# ---------------------------------------------------------------
# Page setup (this must be the first streamlit command)
# ---------------------------------------------------------------
st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")

# ---------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------
# Pydantic models (same as your code)
# ---------------------------------------------------------------
class JobDesc(BaseModel):
    department: str
    responsibilities: str
    technical_skills: str
    education_level: str


class MatchResult(BaseModel):
    score: float
    details: dict


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    address: str | None = None
    phone_number: str | None = None
    experience: float | None = None
    skills: list[str] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


# ---------------------------------------------------------------
# LLM helper: one function for all JSON calls
# ---------------------------------------------------------------
def ask_llm(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------
# Step 1: job description -> structured job
# st.cache_data = same JD text par dobara LLM call nahi hogi
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_job(jd_text: str) -> dict:
    schema = JobDesc.model_json_schema()
    system_prompt = f"""
    You are an expert HR assistant. Extract the job description strictly based
    on the schema provided and give me a JSON output.
    {schema}
    Important: do not return the schema itself. Do not return unnecessary fields.
    Only fill the schema with actual information extracted from the job
    description. Do not invent any information.
    """
    user_prompt = f"Analyse the job description and extract the information:\n{jd_text}"
    data = ask_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return JobDesc(**data).model_dump()


# ---------------------------------------------------------------
# Step 2: read the uploaded resume files
# (uploaded files are file-like objects, so no file path needed)
# ---------------------------------------------------------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file):
    document = Document(file)
    text = ""
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text += cell.text + "\n"
    return text


def read_resume(file, name):
    name = name.lower()
    if name.endswith(".pdf"):
        return read_pdf(file)
    if name.endswith(".docx"):
        return read_docx(file)
    return None


RESUME_FOLDER = Path("resumes")  # same folder as app.py


def get_folder_resumes():
    """Return list of (filename, path) for pdf/docx files in the resumes folder."""
    if not RESUME_FOLDER.exists():
        return []
    return [
        (p.name, p)
        for p in sorted(RESUME_FOLDER.iterdir())
        if p.suffix.lower() in [".pdf", ".docx"]
    ]


# ---------------------------------------------------------------
# Step 3: resume text -> structured resume
# ---------------------------------------------------------------
def parse_resume(resume_text: str) -> Resume:
    schema = Resume.model_json_schema()
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning, not only based
    on exact section headings. Different resumes may use different headings
    (Experience, Professional Experience, Work History, Employment, Internships).
    Skills may appear in the skills section, work experience, internships or projects.

    Return ONLY valid JSON matching this schema:
    {schema}

    Important rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships as experience.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"Parse the following resume:\n\n{resume_text}"
    data = ask_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return Resume(**data)


# ---------------------------------------------------------------
# Step 4: compare resume with job
# ---------------------------------------------------------------
def final_score(job: dict, resume: Resume) -> MatchResult:
    match_schema = MatchResult.model_json_schema()
    prompt = f"""
    You are an HR recruiter.
    Compare the candidate's resume with the job description.

    JOB DESCRIPTION:
    {json.dumps(job, indent=2)}

    CANDIDATE RESUME:
    {resume.model_dump_json(indent=2)}

    Return JSON matching this schema:
    {match_schema}

    In "details" give:
    1. Candidate name
    2. Matching skills
    3. Missing important skills
    4. Whether experience requirement is met
    5. A short final verdict
    "score" must be the overall match percentage from 0 to 100.

    Keep the response concise and easy to read.
    """
    data = ask_llm([{"role": "user", "content": prompt}])
    return MatchResult(**data)


# ===============================================================
# UI starts here
# ===============================================================
st.title("📄 Resume Matcher")
st.caption("Upload resumes, paste a job description, and get candidates ranked by match score.")

# session_state = results yaad rakhta hai jab page rerun hota hai
if "results" not in st.session_state:
    st.session_state.results = []

# ----- Sidebar: inputs -----
with st.sidebar:
    st.header("1. Job Description")
    jd_text = st.text_area("Paste the job description here", height=300)

    st.header("2. Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    folder_resumes = get_folder_resumes()
    use_folder = st.checkbox(
        f"Also use resumes from the 'resumes' folder ({len(folder_resumes)} found)",
        value=len(folder_resumes) > 0,
        disabled=len(folder_resumes) == 0,
    )

    delay = st.slider(
        "Delay between API calls (seconds)",
        min_value=0,
        max_value=10,
        value=2,
        help="Helps avoid Groq rate limits.",
    )

    run = st.button("Analyze Resumes", type="primary", use_container_width=True)

# ----- Run the pipeline -----
if run:
    # one combined list of (name, file_object) from uploads + folder
    sources = [(f.name, f) for f in uploaded_files]
    if use_folder:
        uploaded_names = {name for name, _ in sources}
        for name, path in folder_resumes:
            if name not in uploaded_names:  # avoid duplicates
                sources.append((name, BytesIO(path.read_bytes())))

    if not jd_text.strip():
        st.warning("Please paste a job description first.")
    elif not sources:
        st.warning("Please upload a resume or enable the resumes folder.")
    else:
        with st.spinner("Reading job description..."):
            job = extract_job(jd_text)

        with st.expander("Extracted job info", expanded=False):
            st.json(job)

        results = []
        progress = st.progress(0.0)
        status = st.empty()

        for i, (fname, f) in enumerate(sources):
            status.write(f"Processing **{fname}** ({i + 1}/{len(sources)})")
            try:
                text = read_resume(f, fname)
                if not text or not text.strip():
                    raise ValueError("No text could be extracted (scanned PDF?)")

                parsed = parse_resume(text)  # LLM call 1
                time.sleep(delay)
                match = final_score(job, parsed)  # LLM call 2
                time.sleep(delay)

                results.append(
                    {
                        "file": fname,
                        "name": parsed.name or fname,
                        "score": match.score,
                        "details": match.details,
                        "parsed": parsed.model_dump(),
                    }
                )
            except Exception as e:
                st.error(f"{fname} failed: {e}")

            progress.progress((i + 1) / len(sources))

        status.empty()
        results.sort(key=lambda r: r["score"], reverse=True)
        st.session_state.results = results

# ----- Show results -----
results = st.session_state.results
if results:
    st.subheader("Ranking")

    df = pd.DataFrame(
        [{"Rank": i + 1, "Name": r["name"], "File": r["file"], "Score (%)": r["score"]}
         for i, r in enumerate(results)]
    )
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score (%)": st.column_config.ProgressColumn(
                "Score (%)", min_value=0, max_value=100, format="%.0f"
            )
        },
    )

    st.download_button(
        "Download ranking as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="resume_ranking.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 2")
        for r in results[:2]:
            st.metric(r["name"], f"{r['score']:.0f}%")
    with col2:
        st.subheader("⚠️ Lowest 2")
        for r in results[-2:]:
            st.metric(r["name"], f"{r['score']:.0f}%")

    st.subheader("Candidate details")
    for r in results:
        with st.expander(f"{r['name']} — {r['score']:.0f}%"):
            tab1, tab2 = st.tabs(["Match details", "Parsed resume"])
            with tab1:
                st.json(r["details"])
            with tab2:
                st.json(r["parsed"])
else:
    st.info("Add a job description and resumes in the sidebar, then click **Analyze Resumes**.")