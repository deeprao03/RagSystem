from app.loaders.pdf_loader import load_pdf_documents
from app.llm.generator import generate_answer_with_ollama
from app.processing.chunker import split_documents_into_chunks
from app.processing.structure_extractor import extract_sections_from_documents
from app.retrieval.hybrid_router import classify_query
from app.retrieval.structured_retriever import structured_section_retrieval
from app.retrieval.vector_store import build_faiss_index, similarity_search


def prepare_hybrid_resources(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents_into_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    sections = extract_sections_from_documents(documents)
    vector_store = build_faiss_index(chunks)
    return {
        "documents": documents,
        "chunks": chunks,
        "sections": sections,
        "vector_store": vector_store,
    }


def retrieve_context(query: str, sections: dict, vector_store, top_k: int = 3):
    route = classify_query(query, sections)
    if route == "structured":
        docs = structured_section_retrieval(query, sections, top_k=top_k)
    else:
        docs = similarity_search(vector_store, query=query, k=top_k)
    return route, docs


def answer_query(query: str, sections: dict, vector_store, top_k: int = 3):
    route, docs = retrieve_context(
        query=query,
        sections=sections,
        vector_store=vector_store,
        top_k=top_k,
    )
    answer = generate_answer_with_ollama(query=query, retrieved_docs=docs)
    return {
        "route": route,
        "answer": answer,
        "sources": docs,
    }
