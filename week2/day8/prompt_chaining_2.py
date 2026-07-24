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
                "role": "system",
                "content": "you're a professional C++ programmer. Write clean, optimized c++ code with short explanations"
            },
            {
                "role": "user",
                "content" : prompt
            }
        ]
    )

    return response.choices[0].message.content


# -------------------------
# Prompt Chaining
# -------------------------

# prompt 1
problem = ask_llm(f"sugeest one easy C++ problem for beginner")

# prompt 2
solution = ask_llm(f"solve this c++ problem: {problem}")

# prompt 3
explanation = ask_llm(f"explain this solution: {solution} in simple language")


# ------------------------
# output
# ------------------------

print(problem)
print("\n")

print(solution)
print("\n")

print(explanation)
print("\n")