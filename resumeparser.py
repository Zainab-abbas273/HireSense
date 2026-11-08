import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api error")

client=Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"
jobdescription="""
Job Title: Data Science Intern

Department: Front Office Support

About Business line/Function: Global Markets – Data & AI Lab

Position Purpose

We are, among other things, building models to improve the service we give to our clients (issuing recommendation, anticipating their needs, bringing the relevant research…), to help traders better understanding and managing their risks or leverage alternative data sources (social media, news, images…) for the benefit of our strategists.

We are looking for candidates with education in data science, who not only have experience in solving complex problems but as well understand how and why the model work the way they do.

They need to be motivated with dealing with large amount of very diverse data and extracting valuable insights out of it.

The right candidate needs to be able to adapt quickly to new challenges, not to be afraid to experiment many times and fail before finding the right solution, challenge themselves with the feedback of the users and they will have the excitement of seeing their work being used in real live by the business.

For Internships, We Are Looking At Duration Of 6 Months And We Are Flexible On The Starting Date (the Earlier The Better!). The Intern Will Participate To The Life Of The LAB And Will Take Ownership Of One Or More Topic. We Have a Great Variety Of Topics, And Some Of The Historical Propositions Included

Prediction of which products are the most likely to be interesting for a given client.
Automated Generation of Market Comment.
Optimal Risk Management of Interest Rates Swap Risk.
Regime disentanglement for financial mixture of experts models.
Generative modelling for model control.
Transformers for quantitative investment strategies.

Based on the skillset & business need, we can select a valuable proposition for you!

Responsibilities

Explore and examine data from multiple diverse data sources.
Conceptual modeling, statistical analysis, predictive modeling and optimization design.
Data cleanup, normalization and transformation.
Hypothesis testing: being able to develop hypothesis and test with careful experiments.
Help build workflows for extraction, transformation and loading of different data from a variety of sources and enable linking them to existing systems and datasets.
Ensure the integrity and security of data.

Technical & Behavioral Competencies

Education in data science, who not only have experience in solving complex problems but as well understand how and why the model work the way they do.
Knowledge of key concepts in Statistics and Mathematics such as Probability Theory, Inference, and Linear Algebra.
Knowledge or experience in Machine Learning procedures and tasks such as Classification, Prediction, and Clustering.
Programming skills in Python and knowledge of common numerical and machine-learning packages (NumPy, scikit-learn, pandas, Keras, TensorFlow, PyTorch, langchain).
Ability to write clear and concise code in python.
Intellectually curious and willing to learn challenging concepts daily.
Involvement with the Data Science community through platforms such as Kaggle, Numerai, Open ML, or others.
Knowledge of current Machine Learning/Artificial Intelligence literature.

Technical Skills

Skills Referential (Required knowledge, skills and abilities)

Python, Git
Statistics, ML & NN model architectures
langchain / unsloth / huggingface
Pytorch, Sklearn
Behavioral Skills

Good communication skills
Critical thinking
Ability to Collaborate / Teamwork
Attention to detail / rigor
Education Level: Bachelor’s or Master’s in any of the STEM fields

Location: Bangalore

About BNP Paribas Group

BNP Paribas is the European Union’s leading bank and key player in international banking. It operates in 65 countries and has nearly 185,000 employees, including more than 145,000 in Europe. The Group has key positions in its three main fields of activity: Commercial, Personal Banking & Services for the Group’s commercial & personal banking and several specialized businesses including BNP Paribas Personal Finance and Arval; Investment & Protection Services for savings, investment, and protection solutions; and Corporate & Institutional Banking, focused on corporate and institutional clients. Based on its strong diversified and integrated model, the Group helps all its clients (individuals, community associations, entrepreneurs, SMEs, corporates and institutional clients) to realize their projects through solutions spanning financing, investment, savings and protection insurance. In Europe, BNP Paribas has four domestic markets: Belgium, France, Italy, and Luxembourg. The Group is rolling out its integrated commercial & personal banking model across several Mediterranean countries, Turkey, and Eastern Europe. As a key player in international banking, the Group has leading platforms and business lines in Europe, a strong presence in the Americas as well as a solid and fast-growing business in Asia-Pacific. BNP Paribas has implemented a Corporate Social Responsibility approach in all its activities, enabling it to contribute to the construction of a sustainable future, while ensuring the Group's performance and stability.

About BNP Paribas India Solutions

Established in 2005, BNP Paribas India Solutions is a wholly owned subsidiary of BNP Paribas SA, European Union’s leading bank with an international reach. With delivery centers located in Bengaluru, Chennai and Mumbai, we are a 24x7 global delivery center. India Solutions services three business lines: Corporate and Institutional Banking, Investment Solutions and Retail Banking for BNP Paribas across the Group. Driving innovation and growth, we are harnessing the potential of over 10000 employees, to provide support and develop best-in-class solutions.

Commitment to Diversity and Inclusion

At BNP Paribas, we passionately embrace diversity and are committed to fostering an inclusive workplace where all employees are valued, respected and can bring their authentic selves to work. We prohibit Discrimination and Harassment of any kind and our policies promote equal employment opportunity for all employees and applicants, irrespective of, but not limited to their gender, gender identity, sex, sexual orientation, ethnicity, race, colour, national origin, age, religion, social status, mental or physical disabilities, veteran status etc. As a global Bank, we truly believe that inclusion and diversity of our teams is key to our success in serving our clients and the communities we operate in. As a global Bank, we truly believe that inclusion and diversity of our teams is key to our success in serving our clients and the communities we operate in
"""
class jobdesc(BaseModel):
    
    department:str
    responsibilities:str
    technical_skills:str
    education_level:str
    
    
jd_schema=jobdesc.model_json_schema()
system_prompt=f"""
yoo are an expert HR assistant. extract the job description strictly based on the schema provided and give me a json output.
{jd_schema}
important: do not return  the schema itself. do not return unnecessary fields. only fill the schema with actual information extracted from jon description. do not invent any information.
"""
user_prompt=f"""
analyse the job description and extract the following information from the text:
{jobdescription}
"""
message_system={
    "role":"system",
    "content":system_prompt
}
message_user={
    "role":"user",
    "content":user_prompt
}
response_format={
    "type":"json_object"
}

messages=[message_system, message_user]
response=client.chat.completions.create(model=model, messages=messages,response_format=response_format)

ans=response.choices[0].message.content
raw_json=ans

import json
job_data=json.loads(raw_json)
job=jobdesc(**job_data)
print(job.department)
print(job.responsibilities)
print(job.technical_skills)
print(job.education_level)

class Matchresult(BaseModel):
    score:float
    details: dict

class experience(BaseModel):
    company_name:str | None=None
    role:str | None=None
    years_of_experience:str | None=None
    skills:list[str]=[]
    education:str | None=None
    
class resume(BaseModel):
    name:str | None=None
    email:str | None=None
    address:str | None=None
    phone_number:str | None=None
    experience:float | None=None
    skills:list[str]=[]
    education:list[str]=[]
    projects:list[str]=[]
    certifications:list[str]=[]

resume_schema=resume.model_json_schema()
def final_score(job,resume):
    match_schema=Matchresult.model_json_schema()
    prompt = f"""
    You are an HR recruiter.

    Compare the candidate's resume with the job description.

    JOB DESCRIPTION:
    {job.model_dump_json(indent=2)}

    CANDIDATE RESUME:
    {resume.model_dump_json(indent=2)}
    Return JSON matching this schema:

    {match_schema}

    Give me:

    1. Candidate name
    2. Matching skills
    3. Missing important skills
    4. Whether experience requirement is met
    5. Overall match percentage from 0 to 100
    6. A short final verdict

    Keep the response concise and easy to read.
   """
    message={
        "role": "user",
        "content" : prompt
}
    messages=[message]
    response_format={
        "type": "json_object"
}
    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    data = json.loads(response.choices[0].message.content)
    return Matchresult(**data) 

def parse_resume(resume_text):
    system_prompt=f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    message_system={
        "role" : "system",
        "content" : system_prompt
    }
    message_user={
        "role" : "user",
        "content" : user_prompt
    }
    messages=[message_system, message_user]
    response_format={
        "type": "json_object"
    }
    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    r = resume(**data)
    return r
from pypdf import PdfReader
from docx import Document
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file_path):
    document = Document(file_path)
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


def read_resume(file_path):
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    else:
        return None



# lets do it now
resume_folder = Path("resumes")
all_results=[]
for file_path in resume_folder.iterdir():
    #C:\Users\Pratyush\padho_with_pratyush\week1\day5\resumes\abhay resume new - Abhay Singh.pdf
    if file_path.suffix.lower() not in [".pdf", ".docx"]:
        continue
    print("\nProcessing:", file_path.name)
    resume_text = read_resume(file_path)
    parsed_resume=parse_resume(resume_text) # llm call1
    time.sleep(5)
    result = final_score(job, parsed_resume) #llm caLL2
    #score and details
    #acount chtgpt
    # request bhejna shhur krega millions
    #chattgot server jam ho jayega
    time.sleep(5)
    print("Score:", result.score)
    all_results.append({
        "name": parsed_resume.name,
        "score": result.score,
        "details": result.details
    })
all_results.sort(
    key=lambda candidate: candidate["score"],
    reverse=True
)
top_2 = all_results[:2]
worst_2 = all_results[-2:]


print("TOP 2 CANDIDATES")
for candidate in top_2:

    print(
        candidate["name"],
        "-",
        candidate["score"],
        "%"
    )

    print(candidate["details"])

print("LOWEST 2 CANDIDATES")
for candidate in worst_2:

    print(
        candidate["name"],
        "-",
        candidate["score"],
        "%"
    )
    print(candidate["details"])

    










# role="user"



# from pydantic import BaseModel
# class ticket(BaseModel):
#     name:str
#     issue:str
#     email:str
#     phone_number:str
# schema=ticket.model_json_schema()
# response_format={
#     "type":"json_object"
    
# }
# system_prompt=f"""
# extract the personal information strictly based on the schema provided and give me a json output.
# {schema}
# """
# message_system={
#     "role":"system",
#     "content":system_prompt
# }
# text="hello my name is zainab. i have an iphone that stopped working. my address is mumbai. my email is gfh@gmail.com and phone number is 758648"
# prompt=f"""
# this is the customer ticket please extract the following information from the text:
# {text}
# """
# message={"role": role, "content": prompt}
# messages=[message_system, message]
# response=client.chat.completions.create(model=model, messages=messages,response_format=response_format)
# # print(response)
# ans=response.choices[0].message.content
# print(ans)

# # aage kaise padhte h aasaani ke liye
# import json
# raw_json=ans
# data_file=json.loads(raw_json)
# t=ticket(**data_file)

# print(t.name)
# print(t.issue)
# print(t.email)
# print(t.phone_number)