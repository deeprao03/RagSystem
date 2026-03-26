import re


HEADING_NUMBER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+.+")


def _is_heading_line(line: str) -> bool:
    text = line.strip()
    if len(text) < 3 or len(text) > 90:
        return False

    if HEADING_NUMBER_PATTERN.match(text):
        return True

    words = text.split()
    if len(words) > 12:
        return False

    if text.isupper() and any(ch.isalpha() for ch in text):
        return True

    if text.endswith((".", ",", ":", ";", "?", "!")):
        return False

    title_case_words = sum(1 for w in words if w[:1].isupper())
    if title_case_words >= max(2, int(len(words) * 0.7)):
        return True

    return False


def extract_sections_from_documents(documents):
    sections = {}
    current_title = "Introduction"
    current_lines = []
    current_page = documents[0].metadata.get("page", 0) if documents else 0
    seen_titles = {}

    def save_section(title, page, lines):
        content = "\n".join(lines).strip()
        if not content:
            return

        normalized = title.strip() if title.strip() else "Untitled"
        count = seen_titles.get(normalized, 0)
        seen_titles[normalized] = count + 1
        key = normalized if count == 0 else f"{normalized} ({count + 1})"

        sections[key] = {
            "page": page,
            "content": content,
        }

    for doc in documents:
        page = doc.metadata.get("page", 0)
        lines = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()]

        for line in lines:
            if _is_heading_line(line):
                save_section(current_title, current_page, current_lines)
                current_title = line
                current_lines = []
                current_page = page
            else:
                current_lines.append(line)

    save_section(current_title, current_page, current_lines)
    return sections
