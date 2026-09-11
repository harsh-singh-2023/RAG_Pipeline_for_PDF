# PDF RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that lets you chat with your own PDF documents. It loads PDFs, chunks them, embeds them into a vector database, retrieves relevant context for a question, and generates a grounded answer using an LLM — refusing to answer when the document doesn't contain the information.

## Features

- **PDF ingestion** with `PyMuPDFLoader`, tagging each chunk with source filename and page number
- **Recursive text chunking** (`RecursiveCharacterTextSplitter`) with configurable size/overlap
- **Local embeddings** via `sentence-transformers` (`all-MiniLM-L6-v2`) — no API cost for embedding
- **Persistent vector store** using ChromaDB with cosine similarity
- **Idempotent ingestion** — re-running the pipeline on the same document does not create duplicate chunks (content-hash IDs + upsert)
- **Grounded generation** via Groq's LLM API — answers only from retrieved context, and explicitly says when it doesn't know
- **Source citations** — every answer includes which file/page/score it was pulled from
- **Interactive chat loop** in the notebook

## Architecture

```
PDF files
   │
   ▼
PyMuPDFLoader (load per-page documents)
   │
   ▼
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
   │
   ▼
SentenceTransformer (all-MiniLM-L6-v2) → embeddings
   │
   ▼
ChromaDB (persistent, cosine similarity)
   │
   ▼
Query → embed → retrieve top-k → filter by similarity threshold
   │
   ▼
Groq LLM (openai/gpt-oss-20b) + prompt template → grounded answer
```

## Tech Stack

| Component | Tool |
|---|---|
| Document loading | LangChain (`PyMuPDFLoader`, `DirectoryLoader`) |
| Chunking | `langchain-text-splitters` |
| Embeddings | `sentence-transformers` |
| Vector store | `ChromaDB` |
| LLM | Groq API (`openai/gpt-oss-20b`) via `langchain-groq` |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/harsh-singh-2023/RAG_Pipeline_for_PDF.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Groq API key
Sign up free at [console.groq.com](https://console.groq.com) and generate an API key.

### 4. Set your API key
**Locally:**
```bash
export GROQ_API_KEY="your-key-here"
```

**In Google Colab:** use Colab Secrets (🔑 icon in the sidebar) — add a secret named `GROQ_API_KEY`. Never hardcode API keys directly in notebook cells.

### 5. Add your PDFs
Place PDF files in `data/pdf/`, or use the in-notebook upload cell if running in Colab.

### 6. Run
Open `rag_pipeline.ipynb` and run all cells in order. The final cell starts an interactive chat loop — ask questions about your uploaded document.

## Example

```
You: What is the main topic of this document?
Bot: [grounded answer generated from retrieved chunks]

Sources:
  - report.pdf (page 2, score: 0.812)
```

## Project Structure

```
.
├── rag_pipeline.ipynb      # Main notebook
├── requirements.txt        # Python dependencies
├── data/
│   ├── pdf/                 # Drop PDFs here (gitignored)
│   └── vector_store/        # ChromaDB persistence (gitignored)
├── .gitignore
└── README.md
```

## Known Limitations / Roadmap

- No conversation memory yet — each question is handled independently (follow-up questions like "what about its downsides?" won't resolve pronouns)
- No hybrid (keyword + vector) search — pure semantic retrieval can miss exact terms, names, or numbers
- No reranking step after initial retrieval
- No formal evaluation harness (retrieval precision/recall, answer faithfulness)
- Not yet deployed as a standalone web app (currently notebook-only)

Planned improvements: hybrid search with BM25, cross-encoder reranking, `ragas`-based evaluation, and a Streamlit/Gradio front end.

## License

MIT