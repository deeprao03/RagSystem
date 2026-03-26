from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.config import EMBEDDING_MODEL


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_faiss_index(chunks):
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def similarity_search(vector_store, query: str, k: int = 3):
    return vector_store.similarity_search(query, k=k)
