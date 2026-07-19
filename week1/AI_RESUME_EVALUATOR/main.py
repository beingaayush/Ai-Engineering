from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
import os, json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ResumeEvaluation(BaseModel):
    match_percentage:int
    overall_verdict:str
    matched_skills:list[str]
    missing_skills:list[str]
    strengths:list[str]
    weaknesses:list[str]
    improvement_suggestions:list[str]

def read_pdf(path):
    reader=PdfReader(path)
    text=""
    for page in reader.pages:
        t=page.extract_text()
        if t:
            text+=t+"\n"
    return text

resume=read_pdf("resume.pdf")
with open("job_description.txt","r",encoding="utf-8") as f:
    jd=f.read()

system_prompt="""
You are an ATS Resume Evaluator.
Compare the resume with the job description.

Return ONLY valid JSON in this format:
{
 "match_percentage":0,
 "overall_verdict":"",
 "matched_skills":[],
 "missing_skills":[],
 "strengths":[],
 "weaknesses":[],
 "improvement_suggestions":[]
}
"""

user_prompt=f"""Resume:
{resume}

Job Description:
{jd}
"""

response=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    response_format={"type":"json_object"},
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
)

data=json.loads(response.choices[0].message.content)
result=ResumeEvaluation(**data)

print("\n===== Resume Evaluation =====\n")
print(json.dumps(result.model_dump(),indent=4))

print("\n===== Token Usage =====")
print("Prompt Tokens:",response.usage.prompt_tokens)
print("Completion Tokens:",response.usage.completion_tokens)
print("Total Tokens:",response.usage.total_tokens)