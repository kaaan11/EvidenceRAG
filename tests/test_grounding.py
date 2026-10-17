import json
from pathlib import Path

import pytest

from app.rag.grounding import verify_answer


EVIDENCE = [
    {"text": "Su kayıp oranı yüzde 31'den yüzde 22'ye düştü. Proje 86 milyon liraya mal oldu."},
    {"text": "Tramvay günde 21.500 yolcu taşıyor. Kişi başı yeşil alan 5,8 metrekare."},
]


def test_supported_answer_has_no_findings():
    result = verify_answer(
        "Su kaybı %31'den %22'ye düştü [Kanıt 1]; tramvay günde 21500 yolcu taşıyor [Kanıt 2].",
        EVIDENCE,
    )

    assert result == {
        "citations": [1, 2], "invalid_citations": [],
        "unsupported_numbers": [], "question_only_numbers": [],
    }


def test_reports_numbers_missing_from_evidence():
    result = verify_answer("Proje 90 milyon liraya mal oldu ve 86 milyonluk kısmı... [Kanıt 1]", EVIDENCE)

    assert result["unsupported_numbers"] == ["90"]


def test_reports_citations_to_missing_evidence():
    result = verify_answer("Yeşil alan 5,8 metrekare [Kanıt 2] [Kanıt 4].", EVIDENCE)

    assert result["citations"] == [2]
    assert result["invalid_citations"] == [4]


def test_turkish_formats_are_supported_but_question_numbers_warn():
    result = verify_answer(
        "2025 yılında kişi başı yeşil alan 5.8 metrekareydi; günlük yolcu 21.500.",
        EVIDENCE,
        question="2025 yılında kişi başı yeşil alan ne kadardı?",
    )

    assert result["unsupported_numbers"] == []
    assert result["question_only_numbers"] == ["2025"]


@pytest.mark.parametrize("label", ["Kanıt", "Kanit", "Evidence", "KANIT", "kAnIt", "eViDeNcE"])
@pytest.mark.parametrize("number", ["1", "4", "0"])
def test_citation_aliases_share_parsing_and_number_removal(label, number):
    result = verify_answer(f"Değer 22 [ \t{label}\n{number} \t].", [{"text": "Değer 22."}])
    assert result == {
        "citations": [1] if number == "1" else [],
        "invalid_citations": [] if number == "1" else [int(number)],
        "unsupported_numbers": [], "question_only_numbers": [],
    }


def test_citation_occurrences_are_deduplicated_and_sorted():
    result = verify_answer("[Evidence 4] [Kanit 2] [Kanıt 1] [Evidence 2] [Kanıt 4]", EVIDENCE)
    assert result["citations"] == [1, 2]
    assert result["invalid_citations"] == [4]
    assert result["unsupported_numbers"] == []


@pytest.mark.parametrize("text", ["[1]", "Evidence 1", "Kanıt 1", "[Evidence1]", "[Evidence -1]"])
def test_non_protocol_references_are_not_citations(text):
    result = verify_answer(text, [{"text": "Sayı yok."}])
    assert result["citations"] == []
    assert result["invalid_citations"] == []
    assert result["unsupported_numbers"] == ["1"]


@pytest.mark.parametrize("answer,evidence", [
    ("%22'ye", "yüzde 22"), ("18.000", "18000"), ("18000", "18.000"),
    ("7,4", "7.4"), ("7.4", "7,4"),
    ("1,234,567", "1.234.567"), ("1.234.567", "1234567"),
    ("1234567", "1,234,567"), ("5.800", "5800"),
])
def test_documented_number_equivalences(answer, evidence):
    result = verify_answer(answer, [{"text": evidence}])
    assert result["unsupported_numbers"] == []
    assert result["question_only_numbers"] == []


@pytest.mark.parametrize("answer,evidence", [
    ("5.800", "5,8"), ("5,8", "5.800"), ("1,234", "1234"),
    ("5,800", "5800"), ("12,34,567", "1234567"),
    ("1.234,56", "1,234.56"), ("12:30", "12.30"),
])
def test_ambiguous_or_distinct_number_forms_do_not_collapse(answer, evidence):
    assert verify_answer(answer, [{"text": evidence}])["unsupported_numbers"] == [answer]


def test_question_only_unsupported_and_evidence_numbers_stay_separate():
    result = verify_answer("2025, 2026, 2025 ve 2024", [{"text": "Yıl 2024."}], "2025 veya 2024?")
    assert result["question_only_numbers"] == ["2025"]
    assert result["unsupported_numbers"] == ["2026"]


def test_normalized_question_only_numbers_keep_answer_spelling_and_order():
    result = verify_answer("18.000 ve 7.4 ve 18.000", [{"text": "Sayı yok."}], "18000 ve 7,4?")
    assert result["question_only_numbers"] == ["18.000", "7.4"]
    assert result["unsupported_numbers"] == []


def test_number_in_question_and_evidence_is_not_question_only():
    result = verify_answer("2025", [{"text": "2025"}], "2025?")
    assert result["question_only_numbers"] == []
    assert result["unsupported_numbers"] == []


def test_existing_but_wrong_source_number_does_not_claim_semantic_support():
    result = verify_answer("Değer 22. [Kanıt 2]", [{"text": "Değer 22."}, {"text": "İlgisiz metin."}])
    # Membership in the union of evidence texts is not a claim-to-source check.
    assert result == {
        "citations": [2], "invalid_citations": [],
        "unsupported_numbers": [], "question_only_numbers": [],
    }


CASES = json.loads((Path(__file__).parent / "fixtures/answer_contract/cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", [case for case in CASES if "verification" in case], ids=lambda case: case["id"])
def test_answer_contract_fixtures(case):
    assert verify_answer(case["answer"], [{"text": case["evidence"]}], case.get("question", "")) == case["verification"]
