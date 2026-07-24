import os
from dotenv import load_dotenv
from groq import Groq

# Load API Key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "llama-3.3-70b-versatile"


# Helper Function
def ask_llm(prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -------------------------
# Prompt Chaining
# -------------------------

# Prompt 1
summary = ask_llm(
    "Summarize Artificial Intelligence in 50 words."
)

# Prompt 2 (uses output of Prompt 1)
keywords = ask_llm(
    f"Extract 5 important keywords from this summary: {summary}"
)

# Prompt 3 (uses output of Prompt 2)
questions = ask_llm(
    f"Generate 5 interview questions using these keywords: {keywords}"
)


# Output
print("===== SUMMARY =====")
print(summary)

print("\n===== KEYWORDS =====")
print(keywords)

print("\n===== INTERVIEW QUESTIONS =====")
print(questions)