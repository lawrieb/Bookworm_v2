
# bookworm/app_rag.py
import streamlit as st

from openlibrary_client import (
    search_by_author,
    search_by_genre,
    cover_url_from_id,
)
from rag import rag_summary_search


st.set_page_config(page_title="Bookworm – Open Library RAG", layout="wide")
st.title("Bookworm – Open Library AI Search")

st.write(
    """
This is the **RAG-powered** version of Bookworm.

- Data source: [Open Library](https://openlibrary.org) APIs  
- Semantic “Book Summary” search using local embeddings 
"""
)

mode = st.radio(
    "Search by",
    ["Author", "Genre", "Book Summary (AI)"],
    horizontal=True,
)

query = st.text_input("Enter your search term")

top_k = st.slider("Number of results", min_value=5, max_value=30, value=10, step=1)

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        if mode == "Author":
            docs = search_by_author(query, limit=top_k)
            st.subheader(f"Results for author: “{query}”")
            for doc in docs:
                col1, col2 = st.columns([1, 3])
                with col1:
                    cover = cover_url_from_id(doc.get("cover_id"))
                    if cover:
                        st.image(cover, use_container_width=True)
                with col2:
                    st.markdown(f"### {doc.get('title')}")
                    authors = ", ".join(doc.get("authors", []))
                    if authors:
                        st.markdown(f"**Authors:** {authors}")
                    subjects = ", ".join(doc.get("subjects", [])[:8])
                    if subjects:
                        st.markdown(f"**Subjects:** {subjects}")
                    year = doc.get("first_publish_year")
                    if year:
                        st.markdown(f"_First published: {year}_")
                st.markdown("---")

        elif mode == "Genre":
            docs = search_by_genre(query, limit=top_k)
            st.subheader(f"Results for genre/subject: “{query}”")
            for doc in docs:
                col1, col2 = st.columns([1, 3])
                with col1:
                    cover = cover_url_from_id(doc.get("cover_id"))
                    if cover:
                        st.image(cover, use_container_width=True)
                with col2:
                    st.markdown(f"### {doc.get('title')}")
                    authors = ", ".join(doc.get("authors", []))
                    if authors:
                        st.markdown(f"**Authors:** {authors}")
                    subjects = ", ".join(doc.get("subjects", [])[:8])
                    if subjects:
                        st.markdown(f"**Subjects:** {subjects}")
                    year = doc.get("first_publish_year")
                    if year:
                        st.markdown(f"_First published: {year}_")
                st.markdown("---")

        else:  # Book Summary (AI)
            books = rag_summary_search(query, top_k=top_k)
            st.subheader(f"AI summary search results for: “{query}”")

            if not books:
                st.warning("No books with descriptions found. Try another query.")
            else:
                for book in books:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if book.get("cover_url"):
                            st.image(book["cover_url"], width="stretch")
                    with col2:
                        st.markdown(f"### {book.get('title')}")
                        authors = ", ".join(book.get("authors", []))
                        if authors:
                            st.markdown(f"**Authors:** {authors}")
                        subjects = ", ".join(book.get("subjects", [])[:8])
                        if subjects:
                            st.markdown(f"**Subjects:** {subjects}")
                        year = book.get("first_publish_year")
                        if year:
                            st.markdown(f"_First published: {year}_")
                        st.markdown(
                            f"_Match score: {book.get('similarity', 0):.3f}_"
                        )
                        if book.get("description"):
                            short = book["description"]
                            if len(short) > 600:
                                short = short[:600] + "..."
                            st.markdown(short)
                    st.markdown("---")
