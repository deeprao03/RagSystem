import re

from langchain_core.documents import Document


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def structured_section_retrieval(query: str, sections: dict, top_k: int = 3):
    q = _normalize(query)
    scored = []

    for title, data in sections.items():
        title_norm = _normalize(title)
        content_norm = _normalize(data.get("content", ""))
        score = 0

        if title_norm in q:
            score += 5

        title_words = [w for w in title_norm.split() if len(w) > 3]
        for w in title_words:
            if w in q:
                score += 1

        query_words = [w for w in q.split() if len(w) > 4]
        for w in query_words:
            if w in content_norm:
                score += 0.3

        if score > 0:
            scored.append((score, title, data))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    docs = []
    for score, title, data in top:
        docs.append(
            Document(
                page_content=data.get("content", ""),
                metadata={
                    "page": data.get("page", "unknown"),
                    "section_title": title,
                    "retrieval_type": "structured",
                    "score": round(score, 3),
                },
            )
        )

    return docs
