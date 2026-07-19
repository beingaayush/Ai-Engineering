import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# Load API Key
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

# Groq Client
client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"

# ---------------------- Pydantic Schema ----------------------

class Ticket(BaseModel):
    name: str
    email: str
    phone_number: str
    address: str
    issue: str

schema = Ticket.model_json_schema()

# ---------------------- Customer Ticket ----------------------

text = """
My name is Alex.
I have an iPhone which is not working at all.
My address is Jaipur.
My email is abc@yahoo.com.
My telephone number is 98987.
"""

# ---------------------- Prompts ----------------------

system_prompt = f"""
Extract the information from the customer ticket.
Return ONLY a valid JSON object.
Do not return explanations.
Do not return markdown.
Every field in the schema must be present.
Schema: {schema}
"""

user_prompt = f"""
Extract the following customer ticket into the required JSON format.
{text}
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

# ---------------------- LLM Call ----------------------

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format={
        "type": "json_object"
    }
)

# ---------------------- Parse & Validate ----------------------

response_json = json.loads(response.choices[0].message.content)

ticket = Ticket.model_validate(response_json)

# ---------------------- Output ----------------------

print("Validated Pydantic Object:\n")
print(ticket)

print("\nPython Dictionary:\n")
print(ticket.model_dump())