import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

# Create Groq client
client = Groq(api_key=my_api_key)

# Model
model = "llama-3.3-70b-versatile"

system_prompt = "you're a brand manager, who'll sugeests name for my food company brand, the brand name should be of one word, suggest only one name"
user_prompt = "suggest a brand name for my food company"


# System Message
system_message = {
    "role": "system",
    "content": system_prompt
}

# User Message
user_message = {
    "role": "user",
    "content": user_prompt
}


# LLM Call
response = client.chat.completions.create(
    model=model,
    messages=[
        system_message,
        user_message
    ],
    temperature=1
)

# Print only the generated response
print(response.choices[0].message.content)