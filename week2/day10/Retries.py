import os
from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"
prompt = "Do you know Elon Musk?"
# ---------------------------------

@retry(
    stop=stop_after_attempt(3),          # Maximum 3 attempts
    wait=wait_exponential(multiplier=1)  # Wait: 1s -> 2s -> 4s
)
def ask_llm():
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response


response = ask_llm()

print(response.choices[0].message.content)