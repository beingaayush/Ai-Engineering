import os
from  dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)    
model="llama-3.3-70b-versatile"

# step 1 - Create a knowledge Base
knowledge_base={
    "vgu":"Aayush Kumar is a backend dev and Ai Engineer Student at vivekananda global university college in jaipur",
    "age":"Aayush Kumar is 22 year old"
}

# step 2 - Retrieval
def retrieve_info(question):
    question=question.lower()
    if "vgu" in question:
        return knowledge_base["vgu"]
    elif "age" in question:
        return knowledge_base["age"]
    else: return None



def ask_llm(question):
    context=retrieve_info(question)
    response=client.chat.completions.create(
        model=model,
        messages=[
            {
                "role":"user",
                "content":question
            },
            {
                "role":"system",
                "content":f"""Give the answer in only one line. Give all the answers based on this context: {context}"""
            }
        ]
    )
    answer=response.choices[0].message.content
    return answer

question1="how old is Aayush Kumar?"
question2="What is Aayush Kumar's age?"

print("1st answer: ")
print(ask_llm(question1))
print("\n")
print("2nd answer: ")
print(ask_llm(question2))




# Limitation in this RAG version:- 
# qn1 me hai "old" not "age"
# age aur old same meaning toh hai. RAG ko samajhna chahiye na?
# Yes — proper RAG mein exactly ye hona chahiye. Lekin current retrieval RAG ka semantic retrieval nahi hai.

# current retrieval basically:
# Question
#    ↓
# "age" word hai in the knowledge base?
#    ↓
# YES → age document
# NO  → None



# Now Actual RAG:
# Question
#    ↓
# Embedding banao
#    ↓
# Question ka meaning/vector
#    ↓
# Knowledge-base documents ke vectors se similarity
#    ↓
# Most relevant document
#    ↓
# LLM
#    ↓
# Answer
