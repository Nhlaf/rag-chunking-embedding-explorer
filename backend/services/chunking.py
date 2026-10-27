import re

from .similarity import cosine_similarity


def _make_chunk(index: int, text: str, chunk_type: str = "content") -> dict:
    cleaned = text.strip()
    return {
        "id": index,
        "text": cleaned,
        "chunk_type": chunk_type,
        "character_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "sentence_count": len(re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", cleaned)),
    }


def fixed_chunks(text: str, chunk_size: int, overlap: int = 0) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    chunks = []
    start = 0
    while start < len(text):
        part = text[start : start + chunk_size].strip()
        if part:
            chunks.append(_make_chunk(len(chunks) + 1, part))
        start += chunk_size - overlap
    return chunks


def sentence_chunks(text: str, chunk_size: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(_make_chunk(len(chunks) + 1, current))
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(_make_chunk(len(chunks) + 1, current))
    return chunks


def paragraph_chunks(text: str) -> list[dict]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    return [_make_chunk(index, paragraph) for index, paragraph in enumerate(paragraphs, 1)]


def recursive_chunks(text: str, chunk_size: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    parts = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if part.strip()]
    chunks = []
    current = ""
    for part in parts:
        if len(part) > chunk_size:
            if current:
                chunks.append(_make_chunk(len(chunks) + 1, current))
                current = ""
            chunks.extend(
                _make_chunk(len(chunks) + index + 1, part[start : start + chunk_size])
                for index, start in enumerate(range(0, len(part), chunk_size))
            )
            continue
        candidate = f"{current} {part}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(_make_chunk(len(chunks) + 1, current))
            current = part
        else:
            current = candidate
    if current:
        chunks.append(_make_chunk(len(chunks) + 1, current))
    return chunks


def semantic_chunks(text: str) -> list[dict]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]
    if len(sentences) < 2:
        return [_make_chunk(1, text)]
    from .embeddings import embed_texts

    vectors = embed_texts(sentences)
    chunks = []
    current = sentences[0]
    for index, sentence in enumerate(sentences[1:], 1):
        similarity = cosine_similarity(vectors[index - 1], vectors[index])
        candidate = f"{current} {sentence}"
        if similarity < 0.35:
            chunks.append(_make_chunk(len(chunks) + 1, current))
            current = sentence
        else:
            current = candidate
    chunks.append(_make_chunk(len(chunks) + 1, current))
    return chunks


def structured_chunks(text: str, structured_data: list[dict]) -> list[dict]:
    rows = []
    for sheet in structured_data:
        for row in sheet.get("rows", []):
            values = [str(value).strip() for value in row if value is not None and str(value).strip()]
            if values:
                rows.append(" | ".join(values))
    if rows:
        return [_make_chunk(index, row, "row") for index, row in enumerate(rows, 1)]
    return heading_chunks(text)


def _is_heading(line: str) -> bool:
    value = line.strip()
    if not value or len(value) > 120 or value.endswith((".", ",", ";")):
        return False
    if re.match(r"^#{1,6}\s+\S+", value):
        return True
    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S+", value):
        return True
    if value.isupper() and len(value.split()) <= 12:
        return True
    words = value.split()
    return 1 < len(words) <= 10 and value[0].isupper() and not re.search(r"[.!?]", value)


def heading_chunks(text: str, max_size: int = 1200) -> list[dict]:
    lines = text.splitlines()
    headings = [index for index, line in enumerate(lines) if _is_heading(line)]
    if not headings:
        return recursive_chunks(text, max_size)
    sections = []
    if headings[0] > 0:
        sections.append(("Introduction", "\n".join(lines[: headings[0]])))
    for position, start in enumerate(headings):
        end = headings[position + 1] if position + 1 < len(headings) else len(lines)
        title = lines[start].strip()
        body = "\n".join(lines[start:end]).strip()
        if body:
            sections.append((title, body))
    chunks = []
    for title, section in sections:
        if len(section) <= max_size:
            chunks.append(_make_chunk(len(chunks) + 1, section, "section"))
        else:
            parts = recursive_chunks(section, max_size)
            chunks.extend({**part, "id": len(chunks) + index + 1, "chunk_type": "section"} for index, part in enumerate(parts))
    return chunks or recursive_chunks(text, max_size)


def create_chunks(text: str, structured_data: list[dict] | None = None) -> list[dict]:
    if not text.strip():
        raise ValueError("The document has no text to chunk.")
    if structured_data:
        return structured_chunks(text, structured_data or [])
    return heading_chunks(text)
