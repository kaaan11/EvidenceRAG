import re


_CITATION = re.compile(r"\[\s*(?:Kan[ıi]t|Evidence)\s+([0-9]+)\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_DOT_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
_COMMA_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3}){2,}")
_DECIMAL = re.compile(r"\d+[.,]\d+")


def verify_answer(answer, evidence, question=""):
    """Textual warnings, not semantic verification or a reason to block an answer.

    Citation membership and number presence across all evidence do not prove that
    the cited source supports the claim. Question-only numbers warn separately.
    """
    cited = set()

    def remove_citation(match):
        # One parser both records citations and excludes their complete markers
        # from number scanning, including out-of-range citation numbers.
        cited.add(int(match.group(1)))
        return " "

    number_text = _CITATION.sub(remove_citation, answer)
    cited = sorted(cited)
    valid = set(range(1, len(evidence) + 1))

    evidence_numbers = {
        _normalize_number(number)
        for item in evidence
        for number in _NUMBER.findall(item["text"])
    }
    question_numbers = {_normalize_number(number) for number in _NUMBER.findall(question)}
    unsupported_numbers = []
    question_only_numbers = []
    for number in _NUMBER.findall(number_text):
        normalized = _normalize_number(number)
        if normalized in evidence_numbers:
            continue
        findings = question_only_numbers if normalized in question_numbers else unsupported_numbers
        if number not in findings:
            findings.append(number)

    return {
        "citations": [number for number in cited if number in valid],
        "invalid_citations": [number for number in cited if number not in valid],
        "unsupported_numbers": unsupported_numbers,
        "question_only_numbers": question_only_numbers,
    }


def _normalize_number(number):
    """Conservative textual locale rules, documented in README's answer contract.

    A single comma is decimal; a single dot with a valid three-digit group is
    Turkish thousands. Multiple uniform three-digit groups are unambiguous.
    Mixed/malformed separators and time colons are kept, not guessed.
    """
    if _DOT_THOUSANDS.fullmatch(number):
        return number.replace(".", "")
    if _COMMA_THOUSANDS.fullmatch(number):
        return number.replace(",", "")
    if _DECIMAL.fullmatch(number):
        return number.replace(",", ".")
    return number
