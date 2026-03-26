from app.loaders.pdf_loader import get_extracted_text_preview, load_pdf_documents
from app.llm.generator import generate_answer_with_ollama
from app.processing.chunker import chunk_preview, split_documents_into_chunks
from app.processing.structure_extractor import extract_sections_from_documents
from app.retrieval.vector_store import build_faiss_index, similarity_search


def run_step2_demo(pdf_path: str) -> None:
    documents = load_pdf_documents(pdf_path)
    print(f"Loaded {len(documents)} pages from: {pdf_path}")

    preview = get_extracted_text_preview(documents)
    print("\n--- Extracted Text Preview ---\n")
    print(preview if preview.strip() else "No text extracted.")


def run_step3_demo(pdf_path: str) -> None:
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200)

    print(f"Loaded {len(documents)} pages from: {pdf_path}")
    print(f"Created {len(chunks)} chunks")
    print("Chunk config: chunk_size=1000, chunk_overlap=200")

    print("\n--- Sample Chunks ---\n")
    for idx, chunk_text in chunk_preview(chunks, n=3, max_chars=300):
        print(f"Chunk {idx}: {chunk_text}\n")


def run_step4_demo(pdf_path: str) -> None:
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200)

    print(f"Loaded {len(documents)} pages from: {pdf_path}")
    print(f"Created {len(chunks)} chunks")
    print("Building FAISS index...")

    vector_store = build_faiss_index(chunks)

    query = "What does this book say about financial education?"
    results = similarity_search(vector_store, query=query, k=3)

    print("\n--- Similarity Search Results ---\n")
    print(f"Query: {query}\n")

    for idx, doc in enumerate(results, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")[:300]
        page = doc.metadata.get("page", "unknown")
        print(f"Result {idx} (page {page}): {snippet}\n")


def run_step5_demo(pdf_path: str) -> None:
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200)
    vector_store = build_faiss_index(chunks)

    query = "What does this book say about financial education?"
    retrieved_docs = similarity_search(vector_store, query=query, k=3)

    print("\n--- Retrieved Chunks ---\n")
    for idx, doc in enumerate(retrieved_docs, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")[:250]
        page = doc.metadata.get("page", "unknown")
        print(f"Chunk {idx} (page {page}): {snippet}\n")

    print("Generating answer with Ollama...\n")
    answer = generate_answer_with_ollama(query, retrieved_docs)

    print("--- Final Answer ---\n")
    print(answer)


def run_step6_demo(pdf_path: str) -> None:
    documents = load_pdf_documents(pdf_path)
    sections = extract_sections_from_documents(documents)

    print(f"Loaded {len(documents)} pages from: {pdf_path}")
    print(f"Extracted {len(sections)} sections")

    print("\n--- Sample Extracted Sections ---\n")
    for idx, (title, data) in enumerate(sections.items(), start=1):
        snippet = data["content"].replace("\n", " ")[:220]
        print(f"Section {idx}: {title}")
        print(f"  Page: {data['page']}")
        print(f"  Content: {snippet}\n")
        if idx >= 5:
            break


if __name__ == "__main__":
    sample_pdf_path = "data/raw/sample.pdf"
    run_step6_demo(sample_pdf_path)
