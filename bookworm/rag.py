# bookworm/rag.py
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

from openlibrary_client import search_raw, fetch_description, cover_url_from_id


# 1) Load the embedding model once, cached by Streamlit
@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    # Small semantic model
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
        normalize_embeddings=True,  
        show_progress_bar=False,
    )
    return embs


# def rag_summary_search(
#     query: str,
#     top_k: int = 10,
#     candidate_limit: int = 200,
# ) -> List[Dict[str, Any]]:
#     """
#     Full RAG-style summary search:
#     1) Use Open Library search to get candidate docs
#     2) Pull descriptions for each candidate
#     3) Embed descriptions + query
#     4) Rank by similarity and return top_k results
#     """
#     # retrieve candidates
#     docs = search_raw(query, limit=candidate_limit)

#     # filter for docs with descriptions
#     books: List[Dict[str, Any]] = []
#     descriptions: List[str] = []

#     for doc in docs:
#         work_key = doc.get("key")
#         description = fetch_description(work_key)
#         if not description:
#             continue

#         book = {
#             "work_key": work_key,
#             "title": doc.get("title"),
#             "authors": doc.get("author_name", []),
#             "subjects": doc.get("subject", []),
#             "first_publish_year": doc.get("first_publish_year"),
#             "cover_url": cover_url_from_id(doc.get("cover_i")),
#             "description": description,
#         }
#         books.append(book)
#         descriptions.append(description)

#     if not books:
#         return []

#     # embed query + descriptions
#     query_emb = embed_texts([query])  # (1, d)
#     desc_embs = embed_texts(descriptions)  # (n, d)

#     sims = _cosine_similarity_matrix(desc_embs, query_emb)  # (n,1)
#     sims = sims[:, 0]  # flatten to (n,)

#     # rank and slice
#     idx_sorted = np.argsort(sims)[::-1][:top_k]

#     ranked_books: List[Dict[str, Any]] = []
#     for idx in idx_sorted:
#         b = books[idx].copy()
#         b["similarity"] = float(sims[idx])
#         ranked_books.append(b)

#     return ranked_books

def rag_summary_search(
    query: str,
    top_k: int = 10,
    candidate_limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Full RAG-style summary search (generic):
    1) Use Open Library search to get a larger candidate pool
    2) Expand retrieval using important keywords from the query
    3) Pull descriptions for each candidate
    4) Embed title+description + query
    5) Rank by similarity and return top_k results
    """

    # ---------- 1: primary retrieval ----------
    docs: List[Dict[str, Any]] = search_raw(query, limit=candidate_limit)

    # ---------- 2: generic keyword expansion ----------
    # Take the query, strip out stopwords, and use remaining tokens
    # to pull additional candidates.
    stopwords = {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "at",
        "on", "for", "with", "about", "into", "by", "from", "as",
        "is", "are", "was", "were", "be", "been", "being", "that",
        "this", "these", "those", "it", "its", "there", "here",
        "who", "what", "when", "where", "why", "how",
    }

    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in query.split()]
    keywords = [t for t in tokens if t and t not in stopwords]

    # Only keep a few distinct keywords to avoid too many calls
    keywords = list(dict.fromkeys(keywords))[:5]  # de-dup and cap at 5

    for kw in keywords:
        more = search_raw(kw, limit=80)
        docs.extend(more)

    # ---------- 3: de-duplicate docs by work key ----------
    seen_keys = set()
    unique_docs: List[Dict[str, Any]] = []
    for d in docs:
        key = d.get("key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_docs.append(d)

    # ---------- 4: keep only docs with descriptions ----------
    books: List[Dict[str, Any]] = []
    texts_for_embedding: List[str] = []

    for doc in unique_docs:
        work_key = doc.get("key")
        description = fetch_description(work_key)
        if not description:
            continue

        title = doc.get("title", "")
        # Combine title + description - richer semantics
        text = (title + ". " + description).strip()

        book = {
            "work_key": work_key,
            "title": title,
            "authors": doc.get("author_name", []),
            "subjects": doc.get("subject", []),
            "first_publish_year": doc.get("first_publish_year"),
            "cover_url": cover_url_from_id(doc.get("cover_i")),
            "description": description,
        }
        books.append(book)
        texts_for_embedding.append(text)

    if not books:
        return []

    # Cap for speed
    max_docs = 400
    books = books[:max_docs]
    texts_for_embedding = texts_for_embedding[:max_docs]

    # ---------- 5: embed query + title+description ----------
    query_emb = embed_texts([query])          # (1, d)
    desc_embs = embed_texts(texts_for_embedding)  # (n, d)

    sims = _cosine_similarity_matrix(desc_embs, query_emb)  # (n, 1)
    sims = sims[:, 0]  # flatten to (n,)

    # ---------- 6: rank and slice ----------
    idx_sorted = np.argsort(sims)[::-1][:top_k]

    ranked_books: List[Dict[str, Any]] = []
    for idx in idx_sorted:
        b = books[idx].copy()
        b["similarity"] = float(sims[idx])
        ranked_books.append(b)

    return ranked_books
