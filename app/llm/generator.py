from ollama import Client

from app.utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def build_prompt(query: str, retrieved_docs) -> str:
    context_blocks = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(f"[Chunk {idx} | page {page}]\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are an enterprise document assistant. "
        "Answer only from the provided context. "
        "If the answer is not in the context, clearly say you do not know.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Return a concise and clear answer."
    )
    return prompt


def generate_answer_with_ollama(query: str, retrieved_docs) -> str:
    client = Client(host=OLLAMA_BASE_URL)
    prompt = build_prompt(query, retrieved_docs)

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
    )
    return response["response"].strip()
