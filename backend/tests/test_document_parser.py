from pathlib import Path

from services.document_parser import parse_document


ROOT = Path(__file__).resolve().parents[2]


def test_word_outline_is_classified_from_content():
    result = parse_document(ROOT / "Introduction to Cloud Computing.docx")
    assert result["document_type"] == "structured"
    assert result["extracted_json"]


def test_policy_pdf_is_classified_as_narrative():
    result = parse_document(ROOT / "Employee Information and Company Guidelines.pdf")
    assert result["document_type"] == "unstructured"


def test_empty_context_is_refused():
    from services.llm import REFUSAL, answer_from_context

    assert answer_from_context("What is not in this document?", []) == REFUSAL
