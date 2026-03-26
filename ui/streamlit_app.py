from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.loaders.pdf_loader import load_pdf_documents
from app.llm.generator import generate_answer_with_ollama
from app.processing.chunker import split_documents_into_chunks
from app.processing.structure_extractor import extract_sections_from_documents
from app.retrieval.hybrid_router import classify_query
from app.retrieval.structured_retriever import structured_section_retrieval
from app.retrieval.vector_store import build_faiss_index, similarity_search


DATA_RAW_DIR = Path("data/raw")


def _save_uploaded_pdf(uploaded_file) -> Path:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / uploaded_file.name
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def _prepare_indexes(pdf_path: Path):
    documents = load_pdf_documents(str(pdf_path))
    chunks = split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200)
    sections = extract_sections_from_documents(documents)
    vector_store = build_faiss_index(chunks)
    return documents, chunks, sections, vector_store


def _retrieve_context(query: str, sections: dict, vector_store):
    route = classify_query(query, sections)
    if route == "structured":
        docs = structured_section_retrieval(query, sections, top_k=3)
    else:
        docs = similarity_search(vector_store, query=query, k=3)
    return route, docs


st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="📄", layout="wide")
st.title("Hybrid RAG Enterprise Document Assistant")
st.caption("Traditional RAG + Structure-based retrieval")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.sections = {}
    st.session_state.pdf_path = None

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None:
    saved_pdf = _save_uploaded_pdf(uploaded_pdf)
    st.success(f"Uploaded: {saved_pdf.name}")

    if st.button("Process Document"):
        with st.spinner("Processing PDF and building indexes..."):
            documents, chunks, sections, vector_store = _prepare_indexes(saved_pdf)
            st.session_state.vector_store = vector_store
            st.session_state.sections = sections
            st.session_state.pdf_path = saved_pdf

        st.success(
            f"Ready: {len(documents)} pages, {len(chunks)} chunks, {len(sections)} sections"
        )

query = st.text_input("Ask a question about the uploaded PDF")

if query and st.session_state.vector_store is not None:
    route, retrieved_docs = _retrieve_context(
        query,
        st.session_state.sections,
        st.session_state.vector_store,
    )

    st.markdown(f"**Retrieval route:** `{route}`")

    try:
        answer = generate_answer_with_ollama(query, retrieved_docs)
    except Exception as exc:
        answer = (
            "Could not generate LLM answer. Make sure Ollama is installed, running, and model is pulled. "
            f"Details: {exc}"
        )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for idx, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("section_title", "semantic_chunk")
        snippet = doc.page_content.strip().replace("\n", " ")[:450]
        st.markdown(f"**Source {idx}** | page `{page}` | `{source}`")
        st.write(snippet)

elif query and st.session_state.vector_store is None:
    st.warning("Please upload and process a PDF first.")
