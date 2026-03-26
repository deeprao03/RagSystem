from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents_into_chunks(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def chunk_preview(chunks, n: int = 3, max_chars: int = 300):
    previews = []
    for idx, chunk in enumerate(chunks[:n], start=1):
        text = chunk.page_content.strip().replace("\n", " ")
        previews.append((idx, text[:max_chars]))
    return previews
