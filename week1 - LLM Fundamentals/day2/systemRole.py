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

# System Message
system_message = {
    "role": "system",
    "content": "you're my loving girlfriend"
}

# User Message
user_message = {
    "role": "user",
    "content": "i love you babe"
}

# LLM Call
response = client.chat.completions.create(
    model=model,
    messages=[
        system_message,
        user_message
    ]
)

# Print only the generated response
print(response.choices[0].message.content)


# ---------------- NOTES ----------------
# load_dotenv()        -> Loads variables from .env
# os.getenv()          -> Reads the API key
# Groq()               -> Creates the Groq client
# model                -> Specifies which LLM to use
# role                 -> Who sent the message (system/user/assistant)
# content              -> Actual instruction or prompt
# message              -> One dictionary containing role + content
# messages             -> List of all messages sent to the model
# chat.completions.create() -> Sends the request to the LLM
# response             -> Complete response object
# response.choices[0].message.content -> Extracts only the generated text