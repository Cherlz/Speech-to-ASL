"""Gloss -> sign clip lookup (the "motion database").

A lexicon is a directory with an `index.csv`:

    path,gloss,kind,source
    signs/HELLO.npz,HELLO,sign,WLASL video 27172 (signer 12)
    letters/A.npz,A,letter,sign.mt via spoken-to-signed-translation (MIT)

`kind` is `sign` for lexical signs and `letter` for fingerspelling handshapes.
Several directories can be combined; earlier ones win, and so does the first clip of a gloss.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from s2a.gloss_to_pose.pose import SignPose

INDEX_FIELDS = ["path", "gloss", "kind", "source"]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEXICON = ROOT / "assets" / "fingerspelling"
DATA_LEXICON = ROOT / "data" / "lexicon"  # where the lexicon builders write by default


def default_lexicons(extra: list[str | Path] | None = None) -> list[Path]:
    """Extra lexicons first, then the built sign database (if any), then the A-Z fingerspelling."""
    dirs = [Path(d) for d in extra or []]
    if (DATA_LEXICON / "index.csv").exists() and DATA_LEXICON.resolve() not in [d.resolve() for d in dirs]:
        dirs.append(DATA_LEXICON)
    return dirs + [DEFAULT_LEXICON]


@dataclass
class Match:
    gloss: str  # as produced by the model, e.g. X-MY
    kind: str  # "sign" | "fingerspell" | "missing"
    clips: list[Path]
    key: str = ""  # normalized lexicon key, e.g. POSS-1; used for captions


# ASLG-PC12 writes pronouns as X-<word> (sometimes mangled, e.g. X-Y = "their"). Map them to
# index (IX), possessive (POSS) and reflexive (SELF) signs so they are never fingerspelled.
PRONOUNS = {
    "X-I": "IX-1", "X-ME": "IX-1", "X-YOU": "IX-2", "X-HE": "IX-3", "X-SHE": "IX-3", "X-IT": "IX-3",
    "X-WE": "IX-1PL", "X-US": "IX-1PL", "X-THEY": "IX-3PL", "X-THEM": "IX-3PL",
    "X-MY": "POSS-1", "X-YOUR": "POSS-2", "X-HIS": "POSS-3", "X-HER": "POSS-3", "X-ITS": "POSS-3",
    "X-OUR": "POSS-1PL", "X-OURS": "POSS-1PL", "X-Y": "POSS-3PL", "X-THEIR": "POSS-3PL", "X-IRS": "POSS-3PL",
    "X-POSS": "POSS",  # English 's
    "X-MYSELF": "SELF-1", "X-YOURSELF": "SELF-2", "X-HIMSELF": "SELF-3", "X-HERSELF": "SELF-3",
    "X-ITSELF": "SELF-3", "X-OURSELVES": "SELF-1PL", "X-MSELVES": "SELF-3PL", "X-THEMSELVES": "SELF-3PL",
}
# Pointing / possessive / reflexive signs: never fingerspell these.
REFERENTIAL = re.compile(r"^(IX|POSS|SELF)(-|$)")
# Until the lexicon has dedicated pointing clips, fall back to the citation sign of the English pronoun.
ALIASES = {
    "IX-1": ["ME", "I"], "IX-2": ["YOU"], "IX-3": ["HE", "SHE", "IT"], "IX-1PL": ["WE", "US"], "IX-3PL": ["THEY"],
    "POSS-1": ["MY", "MINE"], "POSS-2": ["YOUR"], "POSS-3": ["HIS", "HER"], "POSS-1PL": ["OUR"],
    "POSS-3PL": ["THEIR"], "SELF-1": ["MYSELF"], "SELF-2": ["YOURSELF"], "SELF-3": ["HIMSELF", "HERSELF", "ITSELF"],
}


def normalize_gloss(gloss: str) -> str:
    """Map a model gloss onto a lexicon key (handles ASLG markers such as DESC- and X-)."""
    gloss = "-".join(gloss.upper().split())  # multi-word glosses: "thank you" -> THANK-YOU
    if gloss in PRONOUNS:
        return PRONOUNS[gloss]
    if REFERENTIAL.match(gloss):
        return gloss
    gloss = re.sub(r"^(DESC|X)-", "", gloss)
    return re.sub(r"\d+$", "", gloss) or gloss


class Lexicon:
    def __init__(self, directories: list[str | Path] | None = None):
        self.signs: dict[str, Path] = {}
        self.letters: dict[str, Path] = {}
        for directory in directories or default_lexicons():
            self._load(Path(directory))

    def _load(self, directory: Path) -> None:
        index = directory / "index.csv"
        if not index.exists():
            raise FileNotFoundError(f"no index.csv in lexicon directory {directory}")
        with open(index, newline="") as f:
            for row in csv.DictReader(f):
                table = self.letters if row["kind"] == "letter" else self.signs
                table.setdefault("-".join(row["gloss"].upper().split()), directory / row["path"])

    def __contains__(self, gloss: str) -> bool:
        return normalize_gloss(gloss) in self.signs

    def lookup(self, gloss: str, fingerspell: bool = True) -> Match:
        key = normalize_gloss(gloss)
        for candidate in [key, *ALIASES.get(key, [])]:
            if candidate in self.signs:
                return Match(gloss, "sign", [self.signs[candidate]], key)
        # Fingerspell words with no known sign (names, rare words, ...).
        letters = [c for c in key if c.isalpha()]
        if fingerspell and letters and not REFERENTIAL.match(key) and all(c in self.letters for c in letters):
            return Match(gloss, "fingerspell", [self.letters[c] for c in letters], key)
        return Match(gloss, "missing", [], key)


def add_entry(directory: str | Path, pose: SignPose, gloss: str, kind: str = "sign", source: str = "") -> Path:
    """Save a clip into a lexicon directory and append it to index.csv."""
    directory = Path(directory)
    sub = "letters" if kind == "letter" else "signs"
    (directory / sub).mkdir(parents=True, exist_ok=True)
    gloss = "-".join(gloss.upper().split())
    name = re.sub(r"[^A-Za-z0-9_-]", "_", gloss)
    rel = Path(sub) / f"{name}.npz"
    i = 1
    while (directory / rel).exists():
        rel, i = Path(sub) / f"{name}_{i}.npz", i + 1
    pose.save(directory / rel)

    index = directory / "index.csv"
    new = not index.exists()
    with open(index, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"path": rel.as_posix(), "gloss": gloss, "kind": kind, "source": source})
    return directory / rel
