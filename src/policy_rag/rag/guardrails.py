"""Input validation and prompt-injection screening."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from policy_rag.models import Chunk

NO_ANSWER_TEXT = (
    "Não encontrei essa informação na base de políticas. "
    "Se a dúvida persistir, entre em contato com o RH."
)

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions",
        r"ignore\s+(as\s+)?instru[cç][oõ]es\s+(anteriores|acima)",
        r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"system\s*prompt",
        r"\bjailbreak\b",
        r"reveal\s+(your|the)\s+(system|hidden)\s+prompt",
        r"<\s*/?\s*(system|assistant|instructions?)\s*>",
        r"\bDAN\s+mode\b",
    )
]


class QuestionTooLongError(ValueError):
    pass


class EmptyQuestionError(ValueError):
    pass


def validate_question(question: str, max_chars: int) -> str:
    cleaned = " ".join(question.split())
    if not cleaned:
        raise EmptyQuestionError("question must not be empty")
    if len(cleaned) > max_chars:
        raise QuestionTooLongError(f"question exceeds {max_chars} characters")
    return cleaned


def detect_injection(text: str) -> list[str]:
    """Return the list of injection patterns matched in `text` (empty when clean)."""
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]


@dataclass
class ScreeningResult:
    safe: list[Chunk]
    flagged: dict[str, list[str]] = field(default_factory=dict)


def screen_chunks(chunks: list[Chunk]) -> ScreeningResult:
    """Drop chunks containing injection-like lines; report what was flagged."""
    result = ScreeningResult(safe=[])
    for chunk in chunks:
        hits = detect_injection(chunk.text)
        if hits:
            result.flagged[chunk.chunk_id] = hits
        else:
            result.safe.append(chunk)
    return result
