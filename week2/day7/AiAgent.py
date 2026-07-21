import os
import json
from dotenv import load_dotenv
from groq import Groq

# -------------------------
# Load API Key
# -------------------------
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama-3.3-70b-versatile"


# -------------------------
# Tool
# -------------------------
def calculator(expression):
    return str(eval(expression))


# -------------------------
# Tool Definition
# -------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# -------------------------
# User Query
# -------------------------
query = input("Ask : ")


# -------------------------
# First LLM Call
# -------------------------
messages = [
    {
        "role": "system",
        "content": (
            "You are an AI Agent.\n"
            "If calculation is required, use the calculator tool.\n"
            "Otherwise answer normally."
        )
    },
    {
        "role": "user",
        "content": query
    }
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

assistant_message = response.choices[0].message


# -------------------------
# Tool Call
# -------------------------
if assistant_message.tool_calls:

    tool_call = assistant_message.tool_calls[0]

    function_name = tool_call.function.name

    arguments = json.loads(tool_call.function.arguments)

    print("\nThought : I should use Calculator")

    print("Action  :", function_name)

    observation = calculator(arguments["expression"])

    print("Observation :", observation)

    # Send tool result back
    messages.append(assistant_message)

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": observation
        }
    )

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    print("\nFinal Answer :")
    print(final_response.choices[0].message.content)

else:

    print("\nFinal Answer :")
    print(assistant_message.content)