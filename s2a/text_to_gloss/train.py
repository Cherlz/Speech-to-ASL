"""Train the English -> ASL gloss Transformer.

    python -m s2a.text_to_gloss.train --data data/aslg --out models/text2gloss
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from s2a.text_to_gloss.data import Batches, read_split
from s2a.text_to_gloss.evaluate import score
from s2a.text_to_gloss.model import ModelConfig, TextToGlossTransformer
from s2a.text_to_gloss.tokenizer import GlossTokenizer


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def translate_batch(model, tokenizer, sentences, device, batch_size=128) -> list[str]:
    model.eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        ids = [tokenizer.encode(s)[: model.config.max_len] for s in sentences[i: i + batch_size]]
        src = torch.full((len(ids), max(map(len, ids))), tokenizer.pad_id, dtype=torch.long)
        for row, seq in enumerate(ids):
            src[row, : len(seq)] = torch.tensor(seq)
        hyps = model.greedy(src.to(device), tokenizer.bos_id, tokenizer.eos_id)
        out += [tokenizer.decode_gloss(h) for h in hyps]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory with {train,dev}.{en,gloss}")
    p.add_argument("--out", required=True, help="output directory for tokenizer + checkpoint")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=4096, help="tokens per batch")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many updates (0 = no limit)")
    p.add_argument("--dev-size", type=int, default=1000, help="dev sentences scored after each epoch")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    train_src, train_tgt = read_split(args.data, "train")
    dev_src, dev_tgt = read_split(args.data, "dev")
    dev_src, dev_tgt = dev_src[: args.dev_size], dev_tgt[: args.dev_size]
    print(f"train={len(train_src)} dev={len(dev_src)} device={device}")

    tokenizer = GlossTokenizer.train(train_src + train_tgt, args.vocab_size)
    tokenizer.save(out / "tokenizer.json")
    config = ModelConfig(vocab_size=len(tokenizer), d_model=args.d_model, heads=args.heads,
                         encoder_layers=args.layers, decoder_layers=args.layers, ff_dim=args.ff_dim,
                         dropout=args.dropout, pad_id=tokenizer.pad_id)
    model = TextToGlossTransformer(config).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    batches = Batches(tokenizer, train_src, train_tgt, max_tokens=args.max_tokens)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    # Linear warmup then inverse-square-root decay (standard for Transformers).
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / args.warmup, math.sqrt(args.warmup / (s + 1))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=args.label_smoothing)

    best, step, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, tokens = time.time(), 0.0, 0
        for src, tgt_in, tgt_out in batches:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            n = int(tgt_out.ne(tokenizer.pad_id).sum())
            total, tokens, step = total + loss.item() * n, tokens + n, step + 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} loss {total / tokens:.3f} lr {schedule.get_last_lr()[0]:.2e}",
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        hyps = translate_batch(model, tokenizer, dev_src, device)
        metrics = score(hyps, dev_tgt)
        metrics.update(epoch=epoch, step=step, train_loss=total / max(tokens, 1), minutes=(time.time() - start) / 60)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        for s, h, r in list(zip(dev_src, hyps, dev_tgt))[:3]:
            print(f"  EN:  {s}\n  HYP: {h}\n  REF: {r}")
        if metrics["bleu"] > best:
            best = metrics["bleu"]
            torch.save(model.checkpoint(), out / "model.pt")
            print(f"  saved best model (dev BLEU {best:.2f})")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
