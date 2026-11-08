import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, VectorParams

from backend.settings import EMBEDDING_DIM, EMBEDDING_MODEL

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

# EMBEDDING_MODEL / EMBEDDING_DIM come from .env (backend/settings.py).

# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

def delete_session_collections(session_id: str) -> None:
    """Drop every Qdrant collection that belongs to a session (old and new prefix)."""
    suffix = session_id.replace('-', '_')
    for name in (f"papeer_{suffix}", f"scholarlens_{suffix}"):
        if qdrant_client.collection_exists(name):
            qdrant_client.delete_collection(name)


def get_collection_name(session_id: str) -> str:
    suffix = session_id.replace('-', '_')
    # Sessions created before the rename to ScholarLens keep their old "papeer_" collection.
    legacy = f"papeer_{suffix}"
    if qdrant_client.collection_exists(legacy):
        return legacy
    return f"scholarlens_{suffix}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str, on_progress=None, batch_size: int = 16) -> None:
    """Embed and store chunks in batches; on_progress(done, total) is called after each batch."""
    store = get_vectorstore(session_id)
    total = len(docs)
    for start in range(0, total, batch_size):
        store.add_documents(docs[start:start + batch_size])
        if on_progress:
            on_progress(min(start + batch_size, total), total)


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def delete_paper(session_id: str, title: str) -> int:
    """Delete every chunk of one document from the session's collection. Returns chunks removed.

    Matches on the stored metadata title client-side (scroll + delete by id), so it works
    without a payload index on Qdrant Cloud.
    """
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return 0
    ids = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name, with_payload=True, limit=256, offset=offset
        )
        ids.extend(p.id for p in points if (p.payload or {}).get("metadata", {}).get("title") == title)
        if offset is None:
            break
    if ids:
        qdrant_client.delete(collection_name=collection_name, points_selector=PointIdsList(points=ids))
    return len(ids)


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)
