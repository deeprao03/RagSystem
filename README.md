# Hybrid RAG Enterprise Document Assistant

A modular enterprise document QA assistant that combines:

- Traditional RAG (embeddings + FAISS similarity search)
- Structure-based retrieval (vectorless section lookup)
- Query routing to choose the right retrieval path

This project is designed for practical document intelligence workflows and interview-ready architecture explanations.

## Features

- Upload and parse PDF documents
- Chunk long text for semantic retrieval
- Build embeddings and FAISS vector index
- Extract document sections/headings for structure-based retrieval
- Route user queries to `structured` or `semantic` retrieval
- Generate grounded answers with LLM (Ollama local model)
- Show source snippets and metadata in Streamlit UI

## Tech Stack

- Python 3.10+
- LangChain + LangChain Community
- HuggingFace sentence-transformer embeddings
- FAISS (local vector index)
- Ollama (local LLM runtime)
- Streamlit (web UI)

## Project Structure

```text
.
├── app/
│   ├── loaders/
│   │   └── pdf_loader.py
│   ├── processing/
│   │   ├── chunker.py
│   │   └── structure_extractor.py
│   ├── retrieval/
│   │   ├── hybrid_router.py
│   │   ├── structured_retriever.py
│   │   └── vector_store.py
│   ├── llm/
│   │   └── generator.py
│   ├── utils/
│   │   └── config.py
│   └── pipeline.py
├── data/
│   ├── raw/
│   └── faiss_index/
├── ui/
│   └── streamlit_app.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd "Rag System"
```

### 2) Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Environment variables

Copy `.env.example` to `.env` and update if needed:

```env
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 5) Optional: Ollama setup for local LLM answering

```bash
ollama serve
ollama pull llama3
```

If Ollama is not available, retrieval still works and sources are shown, but LLM answer generation will fail gracefully.

## Usage

### CLI demo mode

1. Put a PDF in `data/raw/` (example: `data/raw/sample.pdf`)
2. Run:

```bash
.venv/bin/python main.py
```

### Streamlit app

```bash
.venv/bin/streamlit run ui/streamlit_app.py
```

Then:

1. Upload PDF
2. Click **Process Document**
3. Ask question
4. View answer + sources + retrieval route

## Architecture

```text
PDF -> Loader -> Chunker -> (Embeddings -> FAISS)
                  \-> Structure Extractor (sections)

User Query -> Query Router -> Structured Retriever OR Semantic Retriever
                          -> Context -> LLM -> Final Answer + Sources
```

## Hybrid RAG Explanation

### Traditional RAG

- Uses embeddings and vector similarity (FAISS)
- Best for semantic/natural-language questions

### Vectorless (Structure-based) RAG

- Uses headings/sections/page-level structure
- Best for explicit section/chapter/page reference questions

### Hybrid RAG (this project)

- Classifies each query as `structured` or `semantic`
- Chooses the matching retriever
- Improves robustness across mixed enterprise question styles

## Current Limitations

- Heading extraction is heuristic and can include false positives on noisy PDFs
- No persistent FAISS save/load yet (in-memory index per processing run)
- Router is rule-based (not ML classifier)
- Local LLM response quality depends on installed model and hardware

## Future Improvements

- Persist FAISS index to disk and reload quickly
- Add reranking for better source ordering
- Improve structure extraction with layout-aware parsing
- Add hybrid merge + weighted scoring from both retrievers
- Add evaluation metrics (retrieval precision, answer faithfulness)
- Add cloud LLM/API fallback when Ollama is unavailable

## Demo/Interview Talking Points

- Why hybrid retrieval beats pure vector search for enterprise docs
- Query router design and trade-offs
- How retrieval grounding reduces hallucinations
- How modular architecture supports maintainability and scaling
