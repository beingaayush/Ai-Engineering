from sentence_transformers import SentenceTransformer
import chromadb

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector database
client = chromadb.Client()

collection = client.create_collection("documents")

# PDF se aaye chunks (abhi manually)
documents = [
    "Alex is 21 years old.",
    "Rahul is studying computer science.",
    "Priya lives in Jaipur.",
    "John likes playing football."
]

# Add documents to vector DB
collection.add(
    documents=documents,
    ids=["1", "2", "3", "4"]
)

# User query
query = "Alex kitne saal ka hai?"

# Semantic search
results = collection.query(
    query_texts=[query],
    n_results=1
)

print(results["documents"])