"""ASL gloss -> pose sequence: look each gloss up in the motion database and stitch the clips."""
from __future__ import annotations

from s2a.gloss_to_pose.concatenate import concatenate
from s2a.gloss_to_pose.lexicon import Lexicon, Match
from s2a.gloss_to_pose.pose import SignPose

SIGN_SECONDS = 1.0  # cap for a lexical sign
LETTER_SECONDS = 0.35  # cap for one fingerspelled letter
PUNCTUATION = set(".,!?;:'\"")


def gloss_to_pose(gloss: str, lexicon: Lexicon, fingerspell: bool = True) -> tuple[SignPose, list[Match]]:
    tokens = [g.strip("".join(PUNCTUATION)) or g for g in gloss.split()]
    matches = [lexicon.lookup(g, fingerspell) for g in tokens if not set(g) <= PUNCTUATION]
    clips, caps, labels = [], [], []
    for m in matches:
        cap = SIGN_SECONDS if m.kind == "sign" else LETTER_SECONDS
        for i, path in enumerate(m.clips):
            clips.append(SignPose.load(path))
            caps.append(cap)
            labels.append(m.key if m.kind == "sign" else f"{m.key} ({path.stem[0]})")
    return concatenate(clips, caps, labels=labels), matches
