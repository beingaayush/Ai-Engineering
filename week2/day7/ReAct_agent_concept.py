import re


# -----------------------------
# TOOLS
# -----------------------------
def calculator(expression):
    """Calculator Tool"""
    try:
        return eval(expression)
    except Exception:
        return "Invalid Expression"


def weather(city):
    """Weather Tool (Mock API)"""
    weather_data = {
        "jaipur": "34°C, Cloudy",
        "delhi": "37°C, Sunny",
        "mumbai": "29°C, Rainy"
    }

    return weather_data.get(city.lower(), "Weather not available")


def student_marks(name):
    """Database Tool"""
    database = {
        "aayush": 89,
        "rahul": 76,
        "priya": 95
    }

    return database.get(name.lower(), "Student Not Found")


# -----------------------------
# AGENT
# -----------------------------
def agent(user_query):

    print(f"\nUser : {user_query}\n")

    # ---------- Calculator ----------
    if re.search(r"[0-9]+\s*[\+\-\*/]\s*[0-9]+", user_query):

        expression = re.search(
            r"[0-9]+\s*[\+\-\*/]\s*[0-9]+",
            user_query
        ).group()

        print("Thought:")
        print("This question requires calculation.\n")

        print("Action:")
        print("Calling Calculator...\n")

        observation = calculator(expression)

        print("Observation:")
        print(observation)

        print("\nFinal Answer:")
        print(observation)

        return

    # ---------- Weather ----------
    if "weather" in user_query.lower():

        city = user_query.lower().replace("weather", "").strip()

        print("Thought:")
        print("Need weather information.\n")

        print("Action:")
        print("Calling Weather Tool...\n")

        observation = weather(city)

        print("Observation:")
        print(observation)

        print("\nFinal Answer:")
        print(f"Current weather in {city.title()} : {observation}")

        return

    # ---------- Student Database ----------
    if "marks" in user_query.lower():

        words = user_query.split()

        name = words[-1]

        print("Thought:")
        print("Need to search database.\n")

        print("Action:")
        print("Searching Student Database...\n")

        observation = student_marks(name)

        print("Observation:")
        print(observation)

        print("\nFinal Answer:")
        print(f"{name}'s Marks : {observation}")

        return

    # ---------- Unknown ----------
    print("Thought:")
    print("No suitable tool available.\n")

    print("Final Answer:")
    print("Sorry, I don't have a tool for this task.")


# -----------------------------
# MAIN
# -----------------------------
while True:

    query = input("\nAsk Anything (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    agent(query)