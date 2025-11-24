# backend/hr_processor/embedder.py

import os
import cohere
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ---- ENV VARS ----
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "my-hr-index-cohere"

if not COHERE_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY or PINECONE_API_KEY in .env")

# ---- Cohere Client ----
co = cohere.Client(COHERE_API_KEY)

# ---- Pinecone Client ----
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,            # Cohere embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)


# ----------------------------------------------------------
# Embedding Functions
# ----------------------------------------------------------
def embed_text(texts: list[str]) -> list[list[float]]:
    """Generate Cohere embeddings for a batch of texts."""
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings


# ----------------------------------------------------------
# Pinecone: Indexing / Upsert
# ----------------------------------------------------------
def upsert_documents(docs):
    """Upsert a list of text documents into Pinecone."""
    vectors = []
    for i, doc in enumerate(docs):
        emb = embed_text([doc.page_content])[0]
        vectors.append({
            "id": f"doc-{i}",
            "values": emb,
            "metadata": {"text": doc.page_content}
        })

    index.upsert(vectors=vectors)
    return len(vectors)


# ----------------------------------------------------------
# Pinecone: Query
# ----------------------------------------------------------
def query_index(query: str, top_k: int = 3):
    """Query Pinecone and return matching chunks."""
    vector = embed_text([query])[0]

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    return results["matches"]
