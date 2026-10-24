# Ye file embedding model aur Qdrant handle karti hai.
# main.py aur evaluate.py, dono yahan ke search() function ko use karenge.

# -------------------------------------------------------------------------




import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer

# .env file se environment variables load karta hai
load_dotenv()

# Current folder aur knowledgebase.json ka path
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGEBASE_PATH = BASE_DIR / "knowledgebase.json"

# Qdrant Cloud ki credentials .env se milengi
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Qdrant collection aur embedding model ki settings
COLLECTION_NAME = "rag_eval"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384

# Model aur client ko initially empty rakhte hain
_model = None
_client = None


def get_model():
    global _model

    # Model sirf pehli baar load hoga
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def get_client():
    global _client

    # Credentials check karke Qdrant se connection banate hain
    if _client is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError(
                "Set QDRANT_URL and QDRANT_API_KEY in your .env file."
            )

        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return _client


def setup_knowledgebase():
    # Qdrant client aur embedding model obtain karo
    client = get_client()
    model = get_model()

    # JSON file se saare documents load karo
    with open(KNOWLEDGEBASE_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    # Collection nahi hai toh create karo
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,       # Har embedding mein 384 numbers
                distance=Distance.COSINE,  # Similarity compare karne ka method
            ),
        )

        # Category ke basis par filtering fast karne ke liye index
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # Har document ka text nikaalo
    texts = [document["text"] for document in documents]

    # Text ko numerical vectors (embeddings) mein convert karo
    embeddings = model.encode(texts)

    points = []

    # Har document ke liye ID, vector aur original data prepare karo
    for index in range(len(documents)):
        points.append(
            PointStruct(
                id=index + 1,
                vector=embeddings[index].tolist(),
                payload=documents[index],
            )
        )

    # Documents ko Qdrant mein insert/update karo
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    # Total uploaded documents return karo
    return len(points)


def search(query, top_k=3):
    # Qdrant client aur embedding model obtain karo
    client = get_client()
    model = get_model()

    # User ke question ko embedding vector mein convert karo
    query_vector = model.encode(query).tolist()

    # Qdrant se sabse similar top_k documents retrieve karo
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,  # Original document data bhi return karo
    ).points

    # Retrieved documents return karo
    return results