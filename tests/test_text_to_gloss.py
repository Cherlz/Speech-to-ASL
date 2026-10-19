import torch
from torch import nn

from s2a.text_to_gloss.evaluate import score, word_error_rate
from s2a.text_to_gloss.model import ModelConfig, TextToGlossTransformer
from s2a.text_to_gloss.rules import RuleGlosser
from s2a.text_to_gloss.tokenizer import GlossTokenizer
from s2a.text_to_gloss.data import Batches

PAIRS = [
    ("the children went to school .", "CHILD GO TO SCHOOL ."),
    ("my name is ivy .", "X-MY NAME IVY ."),
    ("where is the library ?", "WHERE LIBRARY ?"),
    ("i like pizza .", "X-I LIKE PIZZA ."),
]


def test_tokenizer_roundtrip():
    tok = GlossTokenizer.train([s for p in PAIRS for s in p] * 10, vocab_size=200)
    ids = tok.encode("where is the library ?", add_bos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode_gloss(ids) == "WHERE IS THE LIBRARY ?"


def test_model_can_overfit_tiny_dataset():
    torch.manual_seed(0)
    src, tgt = [s for s, _ in PAIRS], [t for _, t in PAIRS]
    tok = GlossTokenizer.train(src + tgt, vocab_size=200)
    model = TextToGlossTransformer(ModelConfig(vocab_size=len(tok), d_model=64, heads=2, encoder_layers=1,
                                               decoder_layers=1, ff_dim=128, dropout=0.0, pad_id=tok.pad_id))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    batches = Batches(tok, src, tgt, max_tokens=10_000)
    for _ in range(150):
        for s, t_in, t_out in batches:
            loss = loss_fn(model(s, t_in).flatten(0, 1), t_out.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    for s, t in PAIRS:
        x = torch.tensor([tok.encode(s)])
        assert tok.decode_gloss(model.greedy(x, tok.bos_id, tok.eos_id)[0]) == t
        assert tok.decode_gloss(model.beam_search(x, tok.bos_id, tok.eos_id, beam=3)) == t

    restored = TextToGlossTransformer.from_checkpoint(model.checkpoint()).eval()
    x = torch.tensor([tok.encode(PAIRS[0][0])])
    assert restored.greedy(x, tok.bos_id, tok.eos_id) == model.greedy(x, tok.bos_id, tok.eos_id)


def test_metrics():
    assert word_error_rate(["A B C"], ["A B C"]) == 0
    assert word_error_rate(["A C"], ["A B C"]) == 1 / 3
    assert score(["a b"], ["A B"])["exact_match"] == 100


def test_rule_baseline():
    assert RuleGlosser().translate("The children went to the library.") == "CHILD GO LIBRARY"
