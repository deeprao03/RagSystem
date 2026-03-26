from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(pdf_path: str):
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return documents


def get_extracted_text_preview(documents, max_chars: int = 1200) -> str:
    full_text = "\n\n".join(doc.page_content for doc in documents)
    return full_text[:max_chars]
