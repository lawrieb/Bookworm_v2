# Bookworm_v2
Inspired from graduate group project (forked repository bookworm_rec), with added modern components

# Bookworm — AI-Powered Book Recommendation & Semantic Search  
### *RAG (Retrieval-Augmented Generation) using Open Library + Sentence Transformers + Streamlit*

Bookworm is an AI-enhanced book discovery tool that helps users find books using **semantic natural-language search**, traditional search (author/genre), and book metadata from the **Open Library API**.  
This version upgrades the original Bookworm project featuring:

✔ **Retrieval-Augmented Generation (RAG)** style semantic search  
✔ **Sentence Transformer embeddings** for text similarity  
✔ **Open Library summaries as retrieval**  
✔ **Streamlit UI** that runs locally 
✔ Clean, dependency-free setup

This project demonstrates practical AI engineering techniques, including semantic search, vector embeddings, API-based data ingestion, and lightweight app deployment.

## Project Structure
```
.
├── README.md                 # Project documentation
├── LICENSE                   # License for this project
├── requirements.txt          # Python dependencies
│
├── bookworm/                 # Main application package (not fully packaged yet)
│   ├── app.py                # Streamlit UI entry point
│   ├── rag.py                # RAG summary search module
│   └── openlibrary_client.py # Open Library API client
```

---

## Features

### **1. Semantic Book Summary Search (RAG)**  
Users can type natural-language queries such as:

> “a boy wizard who goes to a school for magic, with an evil antagonist”

…and the model returns the **closest matching books**, ranked by semantic similarity.  
The system uses:

- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **Cosine similarity ranking**
- **Dynamic vector store construction via Open Library**

---

### **2. Embedding-Enhanced Retrieval Pipeline**  
Each book is represented by:

- **Title + Description text**
- Encoded into a vector using sentence embeddings  
- Stored in memory and ranked per query  
- Returns top-K similar books

This makes the system resilient and general:  
✔ works with any genre  
✔ supports long or short queries  
✔ works on non-fiction, sci-fi, romance, etc.

---

### **3. Multi-Mode Search UI**
The Streamlit interface supports:

- **Search by Summary (RAG)**
- **Search by Author**
- **Search by Genre / Subjects**
- **Book metadata previews**
- **Book cover thumbnails**

---

### **4. Lightweight, Free, and Fast**
The entire app:

- Runs with **no GPU**
- Uses **free Open Library API**
- Deployment was **$0** on Streamlit Cloud

---

# Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/lawrieb/bookworm_rec.git
cd bookworm_rec
```

## 2. Create a Virtual Environment
```bash
python3 -m venv bookworm_env
source bookworm_env/bin/activate
```

## 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

## 4. Run the App
```bash
streamlit run app.py
```



