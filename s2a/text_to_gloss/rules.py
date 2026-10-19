"""A simple rule-based glosser, kept only as a baseline to compare our model against.

It mimics the reference project's "simple" glosser: drop words ASL usually omits
(articles, forms of "to be", some prepositions), strip common English inflection and upper-case.
"""
from __future__ import annotations

import re

DROP = {"a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being", "of", "to"}
IRREGULAR = {"i": "IX-1", "me": "IX-1", "my": "POSS-1", "you": "IX-2", "your": "POSS-2", "children": "CHILD",
             "went": "GO", "ate": "EAT", "has": "HAVE", "had": "HAVE", "did": "DO", "does": "DO", "people": "PEOPLE"}


def _lemma(word: str) -> str:
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and not word.endswith("ss"):
            return word[: -len(suffix)] + repl
    return word


class RuleGlosser:
    def translate(self, text: str) -> str:
        words = re.findall(r"[A-Za-z']+|\d+", text.lower())
        out = [IRREGULAR.get(w) or _lemma(w).upper() for w in words if w not in DROP]
        return " ".join(out)

    def translate_many(self, texts: list[str]) -> list[str]:
        return [self.translate(t) for t in texts]
