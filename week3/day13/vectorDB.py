# import everything we need
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq


# loading variables from .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# connect to Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud")


# connect to Groq
groq_client = Groq(api_key=GROQ_API_KEY)

print("Connected to Groq")


# create Qdrant collection
COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384


# delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)


# create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    ),
)

print(f"Collection created: {COLLECTION_NAME}")


# Load knowledge.txt
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} documents")


# Create embeddings
print("Loading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Model Ready!")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")


# Create Qdrant points
points = []

for i, (document, embedding) in enumerate(zip(documents, embeddings)):

    point = PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload={
            "text": document
        }
    )

    points.append(point)


# Upload points to Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} points to Qdrant")


# RAG QUERY
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


# PROMPT
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


# CALL GROQ
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