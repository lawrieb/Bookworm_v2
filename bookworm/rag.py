# bookworm/rag.py
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

from openlibrary_client import search_raw, fetch_description, cover_url_from_id


# 1) Load the embedding model once, cached by Streamlit
@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    # Small, widely-used semantic model
    # First run will download it.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (n, d), b: (m, d)
    returns: (n, m) cosine similarity matrix
    """
    # assuming both are already L2-normalized
    return a @ b.T


@st.cache_data(show_spinner=True)
def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine = dot product
        show_progress_bar=False,
    )
    return embs


def rag_summary_search(
    query: str,
    top_k: int = 10,
    candidate_limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Full RAG-style summary search:
    1) Use Open Library search to get candidate docs
    2) Pull descriptions for each candidate
    3) Embed descriptions + query
    4) Rank by similarity and return top_k results
    """
    # Step 1: retrieve candidates
    docs = search_raw(query, limit=candidate_limit)

    # Step 2: filter docs with descriptions
    books: List[Dict[str, Any]] = []
    descriptions: List[str] = []

    for doc in docs:
        work_key = doc.get("key")
        description = fetch_description(work_key)
        if not description:
            continue

        book = {
            "work_key": work_key,
            "title": doc.get("title"),
            "authors": doc.get("author_name", []),
            "subjects": doc.get("subject", []),
            "first_publish_year": doc.get("first_publish_year"),
            "cover_url": cover_url_from_id(doc.get("cover_i")),
            "description": description,
        }
        books.append(book)
        descriptions.append(description)

    if not books:
        return []

    # Step 3: embed query + descriptions
    query_emb = embed_texts([query])  # (1, d)
    desc_embs = embed_texts(descriptions)  # (n, d)

    sims = _cosine_similarity_matrix(desc_embs, query_emb)  # (n,1)
    sims = sims[:, 0]  # flatten to (n,)

    # Step 4: rank and slice
    idx_sorted = np.argsort(sims)[::-1][:top_k]

    ranked_books: List[Dict[str, Any]] = []
    for idx in idx_sorted:
        b = books[idx].copy()
        b["similarity"] = float(sims[idx])
        ranked_books.append(b)

    return ranked_books
