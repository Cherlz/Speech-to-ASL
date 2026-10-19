"""Automatic text-to-gloss metrics, and a CLI to score a model or the rule baseline on a test set.

    python -m s2a.text_to_gloss.evaluate --data data/aslg --split test --model models/text2gloss
    python -m s2a.text_to_gloss.evaluate --data data/aslg --split test --glosser rules

Note: these only measure agreement with the reference gloss. They do not tell us whether
a fluent signer would find the output correct; that needs human evaluation.
"""
from __future__ import annotations

import argparse
import json

import sacrebleu


def word_error_rate(hyps: list[str], refs: list[str]) -> float:
    errors = words = 0
    for h, r in zip(hyps, refs):
        h, r = h.split(), r.split()
        row = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            prev, row[0] = row[0], i
            for j, hw in enumerate(h, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != hw))
        errors, words = errors + row[-1], words + len(r)
    return errors / max(words, 1)


def score(hyps: list[str], refs: list[str]) -> dict:
    hyps, refs = [h.upper() for h in hyps], [r.upper() for r in refs]
    return {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        "wer": round(100 * word_error_rate(hyps, refs), 2),
        "exact_match": round(100 * sum(h == r for h, r in zip(hyps, refs)) / max(len(refs), 1), 2),
    }


def main(argv=None):
    from s2a.text_to_gloss import load_glosser
    from s2a.text_to_gloss.data import read_split

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--glosser", default="model", choices=["model", "rules"])
    p.add_argument("--model", default=None, help="model directory (for --glosser model)")
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    src, refs = read_split(args.data, args.split)
    if args.limit:
        src, refs = src[: args.limit], refs[: args.limit]
    glosser = load_glosser(args.glosser, args.model, beam=args.beam)
    hyps = glosser.translate_many(src)
    print(json.dumps({"glosser": args.glosser, "split": args.split, "n": len(src), **score(hyps, refs)}))


if __name__ == "__main__":
    main()
