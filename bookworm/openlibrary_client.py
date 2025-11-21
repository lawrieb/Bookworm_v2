
# bookworm/openlibrary_client.py
import requests
from typing import List, Dict, Any, Optional

OPENLIB_BASE = "https://openlibrary.org"


def _safe_get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def search_by_author(author: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return a list of works matching an author name."""
    url = f"{OPENLIB_BASE}/search.json"
    data = _safe_get(url, params={"author": author, "limit": limit})

    results = []
    for doc in data.get("docs", []):
        results.append(
            {
                "work_key": doc.get("key"),
                # e.g. "/works/OL12345W"
                "title": doc.get("title"),
                "authors": doc.get("author_name", []),
                "subjects": doc.get("subject", []),
                "first_publish_year": doc.get("first_publish_year"),
                "cover_id": doc.get("cover_i"),
            }
        )
    return results


def search_by_genre(subject: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return works matching a subject/genre."""
    url = f"{OPENLIB_BASE}/search.json"
    data = _safe_get(url, params={"subject": subject, "limit": limit})

    results = []
    for doc in data.get("docs", []):
        results.append(
            {
                "work_key": doc.get("key"),
                "title": doc.get("title"),
                "authors": doc.get("author_name", []),
                "subjects": doc.get("subject", []),
                "first_publish_year": doc.get("first_publish_year"),
                "cover_id": doc.get("cover_i"),
            }
        )
    return results


def search_raw(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Generic text search – used as candidate pool for summary semantic search."""
    url = f"{OPENLIB_BASE}/search.json"
    data = _safe_get(url, params={"q": query, "limit": limit})
    return data.get("docs", [])


def fetch_description(work_key: str) -> Optional[str]:
    """
    Fetch the description/summary for a work.
    work_key is like '/works/OL12345W'.
    """
    if not work_key:
        return None

    url = f"{OPENLIB_BASE}{work_key}.json"
    data = _safe_get(url)

    desc = data.get("description")
    if not desc:
        return None

    if isinstance(desc, str):
        return desc
    if isinstance(desc, dict):
        return desc.get("value")

    return None


def cover_url_from_id(cover_id: Optional[int]) -> Optional[str]:
    if not cover_id:
        return None
    return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
