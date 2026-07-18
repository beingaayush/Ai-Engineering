import os
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model
model = "llama-3.3-70b-versatile"

# Prompts
prompts = [
    "Hi",
    "What is DBMS?",
    "Write a 1000-word essay on Machine Learning."
]

# Loop through prompts
for prompt in prompts:

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens = 2000
    )

    print(f"\nPrompt: {prompt}")
    # print(response.choices[0].message.content)
    print(f"Input: {response.usage.prompt_tokens} | Output: {response.usage.completion_tokens} | Total: {response.usage.total_tokens} Finish reason: {response.choices[0].finish_reason}")