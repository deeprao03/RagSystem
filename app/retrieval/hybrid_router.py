import re


STRUCTURED_HINTS = {
    "section",
    "chapter",
    "clause",
    "heading",
    "title",
    "page",
    "under",
    "according to",
    "where in",
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def classify_query(query: str, sections: dict) -> str:
    q = _normalized(query)

    if any(hint in q for hint in STRUCTURED_HINTS):
        return "structured"

    for title in sections.keys():
        t = _normalized(title)
        if len(t) >= 5 and t in q:
            return "structured"

    numbered_ref = re.search(r"\b\d+(?:\.\d+)+\b", q)
    if numbered_ref:
        return "structured"

    return "semantic"
