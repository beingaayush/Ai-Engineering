# Ye file embedding model aur Qdrant handle karegi. main.py aur evaluate.py,
# dono isi file ke retrieval function ko use karenge.

# -------------------------------------------------------------------------

# retrieval.py

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

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGEBASE_PATH = BASE_DIR / "knowledgebase.json"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "rag_eval"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384

_model = None
_client = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def get_client():
    global _client

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
    client = get_client()
    model = get_model()

    with open(KNOWLEDGEBASE_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=Distance.COSINE,
            ),
        )

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    texts = [document["text"] for document in documents]
    embeddings = model.encode(texts)

    points = []

    for index in range(len(documents)):
        points.append(
            PointStruct(
                id=index + 1,
                vector=embeddings[index].tolist(),
                payload=documents[index],
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)


def search(query, top_k=3):
    client = get_client()
    model = get_model()

    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results