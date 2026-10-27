from services.chunking import create_chunks, fixed_chunks


def test_fixed_chunks_have_expected_sizes():
    chunks = fixed_chunks("abcdefghij", 4)
    assert [chunk["text"] for chunk in chunks] == ["abcd", "efgh", "ij"]


def test_overlap_repeats_boundary_text():
    chunks = fixed_chunks("abcdefghij", 5, 2)
    assert chunks[0]["text"] == "abcde"
    assert chunks[1]["text"].startswith("def")


def test_document_chunking_preserves_similar_sentences_without_size_limit():
    chunks = create_chunks("One sentence. Two sentence. Three sentence.")
    assert len(chunks) == 1
    assert chunks[0]["sentence_count"] == 3


def test_document_chunking_uses_structured_rows_without_size_settings():
    chunks = create_chunks(
        "Name | Team\nAmina | Design\nSam | Engineering",
        [{"name": "Sheet 1", "rows": [["Name", "Team"], ["Amina", "Design"], ["Sam", "Engineering"]]}],
    )

    assert [chunk["text"] for chunk in chunks] == ["Name | Team", "Amina | Design", "Sam | Engineering"]


def test_document_chunking_bounds_oversized_text():
    chunks = create_chunks("word " * 500)

    assert len(chunks) > 1
    assert all(chunk["character_count"] <= 1200 for chunk in chunks)


def test_document_chunking_uses_headings_as_section_boundaries():
    chunks = create_chunks("Introduction\nOverview of the policy.\n\n1. Reporting\nMonthly reports are due on the tenth.\n\n2. Escalation\nContact the manager.")

    assert [chunk["chunk_type"] for chunk in chunks] == ["section", "section", "section"]
    assert chunks[1]["text"].startswith("1. Reporting")
    assert "Monthly reports" in chunks[1]["text"]
