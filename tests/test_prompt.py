from app.rag.prompt import build_messages


def test_messages_require_evidence_only_answers_and_citations():
    system, user = build_messages(
        "What happened?",
        [
            {
                "text": "The event happened in 2024.",
                "filename": "report.pdf",
                "page": 2,
                "chunk_id": "doc-1-0",
            }
        ],
    )

    assert system["role"] == "system"
    assert "Only answer using the supplied evidence" in system["content"]
    assert "reply with only NO_EVIDENCE" in system["content"]
    assert "complete reply" in system["content"]
    assert "[Kanıt N]" in system["content"]
    assert "[Kanit N]" in system["content"]
    assert "[Evidence N]" in system["content"]
    assert "Do not show internal chunk IDs" in system["content"]

    assert user["role"] == "user"
    assert "Kanıt 1" in user["content"]
    assert "report.pdf" in user["content"]
    assert "page=2" in user["content"]
    assert "The event happened in 2024." in user["content"]
    assert "Question: What happened?" in user["content"]
    assert user["content"].endswith("Answer in the same language as the question.")
    assert "doc-1-0" not in user["content"]
