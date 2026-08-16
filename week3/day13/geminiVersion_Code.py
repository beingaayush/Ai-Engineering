# ==========================================
# 1. IMPORT EVERYTHING WE NEED
# ==========================================

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================
# 3. CONNECT TO CLIENTS (Qdrant & Groq)
# ==========================================

# Connect to Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud")

# Connect to Groq
groq_client = Groq(api_key=GROQ_API_KEY)

print("Connected to Groq")


# ==========================================
# 4. INITIALIZE QDRANT COLLECTION (Only If Missing)
# ==========================================

COLLECTION_NAME = "knowledge"

# Sirf tabhi collection banayein jab woh pehle se exist na karti ho
# Isse har query par database delete nahi hoga
if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating collection: {COLLECTION_NAME}")
    
    # Hum seedhe documents list ko metadata ke sath initialize kar rahe hain
    # Isse point creation aur model loop ka complex code nahi likhna padega
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(documents)} documents for initialization")
    
    # Qdrant ka high-level 'add' function background mein automatically
    # vectors configuration, embedding, aur upload handle kar leta hai
    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=[{"text": doc} for doc in documents]
    )
    print("Collection created and documents uploaded successfully!")


# ==========================================
# 5. LOAD EMBEDDING MODEL FOR QUERY
# ==========================================

print("\nLoading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Model Ready!")


# ==========================================
# 6. USER QUERY & SEARCH
# ==========================================

query = input("\nAsk a question: ")

# Convert user query into embedding
query_embedding = model.encode(query).tolist()

# Search relevant documents from Qdrant
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=3
).points

# Extract relevant text
context = "\n".join(
    result.payload["text"]
    for result in search_results
)

print("\nRetrieved Context:")
print(context)


# ==========================================
# 7. GENERATE ANSWER USING GROQ
# ==========================================

prompt = f"""
You are a helpful company policy assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't have enough information to answer that."

Context:
{context}

User Question:
{query}
"""

response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You answer questions using the provided context."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

# Final answer
answer = response.choices[0].message.content

print("\nFinal Answer:")
print(answer)



# diff b/w this version and vectorDB.py version = isme do crucial changes ko adjust kiya hai jo performance aur simplicity ko behtar banayenge:

# client.add() ka use: PointStruct, loop, vectors config, aur vectors deletion wala lamba boilerplate code hata diya.
# Isse formatting bilkul waisi hi rahegi jaisi tumhari thi, bas internal logic chota ho jayega.

# Collection recreate problem solved: Tumhare original flow mein har baar naya sawaal puchne par collection delete ho rahi thi.
# Isko humne ek simple if not client.collection_exists() check se replace kar diya hai, taaki database sirf ek baar bane.