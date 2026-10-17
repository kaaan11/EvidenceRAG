NO_EVIDENCE_MARKER = "NO_EVIDENCE"

SYSTEM_PROMPT = f"""You are an evidence-grounded assistant.
Only answer using the supplied evidence. Do not use outside knowledge or invent facts.
If the evidence does not contain the answer, reply with only {NO_EVIDENCE_MARKER} and nothing else.
The refusal marker must be the complete reply, without an explanation or quotation.
Cite supporting evidence using the canonical form [Kanıt N], for example [Kanıt 1].
Bracketed [Kanit N] and [Evidence N] are accepted aliases; prefer [Kanıt N].
Do not show internal chunk IDs in the answer."""


def build_messages(question, evidence):
    evidence_text = "\n\n".join(
        _format_evidence(index, item) for index, item in enumerate(evidence, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Evidence:\n{evidence_text}\n\nQuestion: {question}\n\n"
                "Answer in the same language as the question."
            ),
        },
    ]


def _format_evidence(index, item):
    page = item.get("page") if item.get("page") is not None else "n/a"
    return (
        f"Kanıt {index} | "
        f"filename={item.get('filename')} | page={page}\n{item['text']}"
    )
