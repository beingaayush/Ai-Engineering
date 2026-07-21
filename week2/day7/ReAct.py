import os
from dotenv import load_dotenv
from groq import Groq

# Load API Key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model
model = "llama-3.3-70b-versatile"

# User Query
user_query = "i wan to purchase an iphone 17, i have 1.5 lacs rupees, after purchasing that item how much money i've left with?"

# ReAct Prompt
prompt = f"""
You are an AI assistant that follows the ReAct framework.

For every question, respond in this format:

Thought:
Explain your reasoning.

Action:
Mention the tool you would use.

Observation:
Write the tool's output.

Final Answer:
Give the final answer.

Question:
{user_query}
"""

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant following the ReAct pattern."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)



# NOTE:-
# This is not a real AI Agent.
# we're only prompting the LLM to pretend it is using tools.
# A real ReAct agent works like this:

#                User
#                 ↓
#                LLM
#                 ↓
#              Chooses Tool
#                 ↓
#       Python/Web Search/Database/API
#                 ↓
#            Gets Real Output
#                 ↓
#                LLM
#                 ↓
#             Final Answer