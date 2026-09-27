# RAG CHUNKING STRATEGIES

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

text = """
RAG retrieves relevant information from documents.
Embeddings convert text into vectors.
Vector databases store these embeddings.

TCP uses a three-way handshake.
TCP provides reliable data delivery.
"""

# 1. FIXED-SIZE CHUNKING
chunk_size = 100

fixed_chunks = []

for i in range(0, len(text), chunk_size):
    fixed_chunks.append(text[i:i + chunk_size])


# 2. FIXED-SIZE + OVERLAP
overlap = 20
overlap_chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    overlap_chunks.append(text[start:end])
    start = end - overlap


# 3. RECURSIVE CHUNKING
recursive_chunks = []

paragraphs = text.split("\n\n")

for paragraph in paragraphs:
    if len(paragraph) <= chunk_size:
        recursive_chunks.append(paragraph.strip())
    else:
        sentences = paragraph.split(". ")

        for sentence in sentences:
            if sentence.strip():
                recursive_chunks.append(sentence.strip())


# 4. SEMANTIC CHUNKING
sentences = [
    s.strip()
    for s in text.split("\n")
    if s.strip()
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

semantic_chunks = []
current_chunk = [sentences[0]]

threshold = 0.5

for i in range(1, len(sentences)):

    similarity = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    if similarity < threshold:
        semantic_chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i]]
    else:
        current_chunk.append(sentences[i])

semantic_chunks.append(" ".join(current_chunk))


# RESULTS
print("Fixed:", fixed_chunks)
print("Overlap:", overlap_chunks)
print("Recursive:", recursive_chunks)
print("Semantic:", semantic_chunks)