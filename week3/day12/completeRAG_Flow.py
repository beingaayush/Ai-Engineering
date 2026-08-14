import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    """
    Alex is 21 years old. He is a computer science student.
    He is currently studying at Vivekananda Global University.
    """,

    """
    Alex lives in Jaipur, Rajasthan. Jaipur is known as the
    Pink City of India.
    """,

    """
    The company offers 20 paid leaves to its employees every year.
    Employees can use these leaves for personal or vacation purposes.
    """,

    """
    Python is a high-level programming language. It is widely used
    for web development, automation, data science and artificial intelligence.
    """
]


# ============================================================
# 2. CHUNKING
# ============================================================

def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = []

for document in documents:
    document_chunks = chunk_text(document)
    chunks.extend(document_chunks)


print("Total chunks:", len(chunks))


# ============================================================
# 3. EMBEDDINGS
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embedding shape:", document_embeddings.shape)


# ============================================================
# 4. VECTOR DATABASE
# ============================================================

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# ============================================================
# 5. USER QUERY
# ============================================================

query = "How old is Alex?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)


# ============================================================
# 6. RETRIEVER
# ============================================================

top_k = 2

distances, indices = index.search(
    query_embedding,
    top_k
)


# ============================================================
# 7. RELEVANT CHUNKS
# ============================================================

relevant_chunks = []

for i in indices[0]:
    relevant_chunks.append(chunks[i])


print("\nRelevant Chunks:")

for chunk in relevant_chunks:
    print("--------------------")
    print(chunk)


# ============================================================
# 8. CREATE CONTEXT
# ============================================================

context = "\n\n".join(relevant_chunks)


# ============================================================
# 9. PROMPT + CONTEXT
# ============================================================

prompt = f"""
Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say:
"I don't know based on the provided context."

Answer:
"""


# ============================================================
# 10. LLM
# ============================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0,
    streaming = True
)


# ============================================================
# 11. FINAL ANSWER
# ============================================================

answer = response.choices[0].message.content

print("\nFinal Answer:")
print(answer)




# complete Flow :
# Documents
#     ↓
# Chunking
#     ↓
# Embeddings
#     ↓
# FAISS Vector Database
#     ↓
# User Query
#     ↓
# Query Embedding
#     ↓
# Retriever
#     ↓
# Relevant Chunks
#     ↓
# Prompt + Context
#     ↓
# Groq LLM
#     ↓
# Final Answer