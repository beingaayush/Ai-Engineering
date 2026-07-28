import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

# ----------------------------------------------------

response = client.chat.completions.create(
    model=model,
    messages=[{
        "role":"user",
        "content":"Explain the steps of making an car"
    }],
    stream=True   # <-- streaming enable
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(
            chunk.choices[0].delta.content,
            end=""
        )

