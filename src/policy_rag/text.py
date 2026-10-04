"""Shared text normalisation helpers (pt-BR aware tokenizer)."""

from __future__ import annotations

import re
import unicodedata

_RAW_STOPWORDS = """
    a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre
    e ou mas que se ao aos à às pelo pela pelos pelas este esta isto esse essa isso aquele
    aquela aquilo seu sua seus suas meu minha meus minhas é são foi ser está estão ter tem
    têm há como quando onde qual quais quanto quantos quem eu tu ele ela nós vós eles elas
    me te lhe nos lhes já não sim também muito mais menos até após entre desde cada
    posso pode podem devo deve devem preciso quero gostaria saber qualquer todo toda todos
    todas outro outra outros outras mesmo mesma vou vai
    """.split()  # noqa: SIM905

_TOKEN = re.compile(r"[a-z0-9]+")


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


STOPWORDS = frozenset(strip_accents(word) for word in _RAW_STOPWORDS)


def stem(token: str) -> str:
    """Very light stemming: cut common pt-BR suffixes and truncate long tokens."""
    for suffix in ("mente", "ções", "coes", "ção", "cao", "oes", "ais", "eis", "res", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token[:7]


def tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, drop stopwords and lightly stem."""
    lowered = strip_accents(text.lower())
    return [stem(tok) for tok in _TOKEN.findall(lowered) if tok not in STOPWORDS and len(tok) > 1]
