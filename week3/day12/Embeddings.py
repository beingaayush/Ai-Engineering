import os
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional vector


documents = [
    "Alex is 21 years old.",
    "Alex lives in Jaipur.",
    "The company offers 20 paid leaves.",
    "Python is a programming language."
]

query = "How old is Alex?"


# Create embeddings for documents
document_embeddings = model.encode(documents)

# Create embedding for query
query_embedding = model.encode(query)


# Calculate cosine similarity
similarities = np.dot(
    document_embeddings,
    query_embedding
) / (
    np.linalg.norm(document_embeddings, axis=1)
    * np.linalg.norm(query_embedding)
)


# Display results
for document, score in zip(documents, similarities):
    print(f"{score:.4f} → {document}")


# Find most similar document
best_index = np.argmax(similarities)

print("\nMost Relevant Document:")
print(documents[best_index])