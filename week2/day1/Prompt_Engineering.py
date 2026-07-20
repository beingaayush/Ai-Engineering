import os
from dotenv import load_dotenv
from groq import Groq

# Load API Key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model
model = "llama-3.3-70b-versatile"

# User Message
user_message = "hello i am 22 year old, gender = male, current weight = 51 kg, wants to gain muscles and make abs"

# Prompt
prompt = f"""
Role:
You are a certified fitness coach.

Task:
Create a weekly workout plan.

Context:
{user_message}

Constraints:
- No gym equipment.
- Workouts should take 30 minutes.
- Include one rest day.

Output Format:
Return the plan as a clean table.

Fallback:
If the user's goal or fitness level is missing, ask for that information before creating the plan.
"""

# LLM Call
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.7
)

print(response.choices[0].message.content)