import json
import os
from groq import Groq

# -----------------------------
# 1. TOOL IMPLEMENTATIONS
# -----------------------------
def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def weather(city: str) -> str:
    """Mock Weather API"""
    weather_data = {
        "jaipur": "34°C, Cloudy",
        "delhi": "37°C, Sunny",
        "mumbai": "29°C, Rainy"
    }
    return weather_data.get(city.lower(), "Weather data not available for this city.")

def student_marks(name: str) -> str:
    """Mock Database Lookup"""
    database = {
        "aayush": 89,
        "rahul": 76,
        "priya": 95
    }
    marks = database.get(name.lower())
    if marks is not None:
        return f"{name.title()} scored {marks} marks."
    return f"Student '{name}' not found in database."

# Map function names to actual Python functions
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "weather": weather,
    "student_marks": student_marks
}

# -----------------------------
# 2. TOOL DEFINITIONS (JSON SCHEMA)
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '89 + 10' or '34 * 2'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Fetches current weather information for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g., 'Jaipur'."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "student_marks",
            "description": "Retrieves the test marks for a student by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The student's first name, e.g., 'aayush'."
                    }
                },
                "required": ["name"]
            }
        }
    }
]

# -----------------------------
# 3. AGENT EXECUTION LOOP
# -----------------------------
client = Groq()  # Automatically picks up GROQ_API_KEY from environment variables
MODEL_NAME = "llama-3.3-70b-versatile"

def agent(user_query: str):
    print(f"\nUser : {user_query}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use tools when necessary to answer user queries accurately."
        },
        {
            "role": "user", 
            "content": user_query
        }
    ]

    while True:
        # Ask Llama 3.3 to evaluate the user query and decide if a tool is needed
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If the LLM doesn't want to call any more tools, break and show the final answer
        if not tool_calls:
            print(f"\nFinal Answer:\n{response_message.content}")
            break

        # Append the assistant's request (including tool call decision) to conversation history
        messages.append(response_message)

        # Execute each tool requested by the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n[Thought]: Needs to execute tool '{function_name}'")
            print(f"[Action] : Calling {function_name} with arguments {function_args}")

            # Call local tool function
            function_to_call = AVAILABLE_TOOLS[function_name]
            tool_output = function_to_call(**function_args)

            print(f"[Observation]: {tool_output}")

            # Pass the result back to Llama 3.3
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(tool_output)
            })

# -----------------------------
# 4. MAIN INTERACTIVE LOOP
# -----------------------------
if __name__ == "__main__":
    print("AI Agent initialized with Llama-3.3-70B-Versatile on Groq!")
    while True:
        query = input("\nAsk Anything (type 'exit' to quit): ")
        if query.lower().strip() == "exit":
            break
        if query.strip():
            agent(query)