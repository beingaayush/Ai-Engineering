import os
from  dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)    

model="llama-3.3-70b-versatile"
prompt="do you know elon musk ?"

response=client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response.choices[0].message.content)








# load_dotenv()          -> Loads variables from .env
# os.getenv()            -> Reads the API key
# OpenAI()               -> Creates the API client
# client.responses.create()  -> Sends request to the LLM
# model                  -> Specifies which LLM to use
# role                   -> Who sent the message (system/user/assistant)
# content                -> Actual text of the message
# response               -> Complete response object returned by the API
# response.output_text   -> Extracts only the generated text