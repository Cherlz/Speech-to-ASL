# S2A: Speech/Text → ASL (low-fidelity prototype)

A `speech → text → gloss → pose → video` pipeline for translating English into American Sign Language.
The structure follows [sign-language-processing/spoken-to-signed-translation](https://github.com/sign-language-processing/spoken-to-signed-translation).
The main difference is the text-to-gloss step. The reference project uses rule-based or hosted "glossers".
**We train our own neural model** for that step.

```
 audio ──► Whisper ──► English text ──► OUR Transformer ──► ASL gloss ──► motion database ──► stitched poses ──► stick-figure video
          (reused)                       (trained here)                   (lexicon lookup +     (normalize, trim,    (OpenCV)
                                                                           fingerspelling)       speed-cap, blend)
```

| Stage | Module | Ours or reused |
|---|---|---|
| Speech → text | `s2a/speech_to_text.py` | Reused (OpenAI Whisper) |
| Text → gloss | `s2a/text_to_gloss/` | **Ours**: BPE tokenizer + Transformer encoder-decoder, trained from scratch |
| Gloss → pose | `s2a/gloss_to_pose/` | Ours. Same idea as the reference: lexicon lookup, fingerspelling fallback, concatenation |
| Video → pose (building the database) | `s2a/gloss_to_pose/extract.py` | MediaPipe Holistic |
| Pose → video | `s2a/pose_to_video.py` | Ours (stick figure). The reference uses pix2pix for photoreal output |

## Install

```bash
pip install -e ".[dev]"            # core: torch, tokenizers, opencv
pip install -e ".[speech]"         # + Whisper, for audio input (also needs ffmpeg)
pip install -e ".[extract]"        # + MediaPipe, for turning sign videos into poses
```

## 1. Train our text → gloss model

```bash
python scripts/prepare_aslg.py                                   # ASLG-PC12 → data/aslg/{train,dev,test}.{en,gloss}
python -m s2a.text_to_gloss.train --data data/aslg --out models/text2gloss --epochs 10
python -m s2a.text_to_gloss.evaluate --data data/aslg --split test --model models/text2gloss
python -m s2a.text_to_gloss.evaluate --data data/aslg --split test --glosser rules   # baseline to compare against
```

- The default model has 7.6M parameters: 3+3 layers, d_model 256, and a shared 8k BPE vocabulary with tied embeddings.
- On a 4-core CPU one epoch takes about 15 minutes. On a GPU it takes a minute or two, and `--device auto` picks the GPU up.
- Useful flags: `--d-model`, `--layers`, `--heads`, `--epochs`, `--max-tokens` and `--max-steps` (for quick experiments).
- The data format is two aligned text files per split. To use a different or better dataset, write it in the same layout.

## Quick demo (no training needed: the trained model is in `models/text2gloss`)

```bash
pip install -e .
s2a demo
```

This opens a page in your browser. Type a sentence, or click **🎤 Speak**, which works in Chrome, Edge and Safari. The page shows the gloss, how each word was covered, and the signing animation. Press `Ctrl+C` in Terminal to stop it.

## 2. Translate

```bash
s2a text-to-gloss  --text "Where is the library?"
s2a text-to-video  --text "My name is Ivy." --video ivy.mp4 --pose ivy.npz
s2a speech-to-video --audio question.wav --video out.mp4
```

- Add `--glosser rules` to use the baseline instead of our model.
- Use `--beam N` to set the beam size, and `--beam 1` for greedy decoding.
- The tool prints how each gloss was covered:
  - green: a real sign from the lexicon
  - yellow: fingerspelled
  - red: missing

## 3. Grow the motion database (shared with A2S)

A lexicon is a directory of `.npz` pose clips plus an `index.csv` (`path,gloss,kind,source`).

- Each clip holds 75 MediaPipe keypoints per frame: 33 body points and 21 for each hand.
- The A2S team can load the same files for recognition (this is Ivy's shared-database idea).
- The repo ships `assets/fingerspelling/`, which has the ASL alphabet A–Z. It was converted from the reference project's MIT-licensed lexicon.

```bash
s2a lexicon-add --lexicon data/lexicon --gloss HELLO --video hello.mp4 --source "WLASL / signer X"
s2a lexicon-add --lexicon data/lexicon --gloss NAME  --pose-file name.pose
s2a text-to-video --text "Hello, my name is Ivy." --lexicon data/lexicon --video out.mp4
```

Until real sign clips are added, **every word is fingerspelled**. Filling `data/lexicon` is the next step that improves the output most. Good sources are WLASL or ASL Citizen videos (check their licenses), or recordings the team makes with fluent signers.

How gloss is matched to clips (`s2a/gloss_to_pose/lexicon.py`):
- Markers are stripped before lookup, so `DESC-FAR` is looked up as `FAR`.
- ASLG pronouns are mapped to index signs, so `X-I` becomes `IX-1`.
- Pointing and possessive glosses (`IX-*`, `POSS-*`, `SELF-*`) are never fingerspelled.

## Tests

```bash
pytest -q
```

The tests cover:
- pose maths
- lexicon lookup and fingerspelling
- a model that overfits a toy dataset (checks the model and decoding code)
- the end-to-end CLI

## Limitations (please read before the design review)

1. **ASLG-PC12 is not real ASL.** Its gloss was generated from English parliament text by rules, not written by signers. Scores on it show the model learns the mapping, not that the output is correct ASL. We need a gloss dataset checked by ASL experts. This is the stakeholder outreach from the 10/12 meeting.
2. **The model learns ASLG's conventions:** `X-` pronouns, `DESC-` adjectives, and a mostly English word order. Its output will look like English with markers until we train on better data.
3. **No non-manual markers.** ASL grammar uses facial expressions, for example for questions and negation. The pose format drops the face. Adding face landmarks is a planned extension.
4. **Stitching citation-form signs is not fluent signing.** It has no role shift, no use of space and no classifiers. It is a feasibility demo.
5. **Automatic metrics (BLEU/WER) are a proxy.** We still need a human evaluation with fluent signers.

## Suggested split of work (S2A team)

- **Data:** look for or build a better English↔gloss dataset (ASL experts, How2Sign, ASL-LEX), and decide on a gloss convention (IX-1/IX-2/IX-3, etc.).
- **Model:** experiments with architecture and hyperparameters, and fine-tuning a pretrained model (such as T5) as a comparison against our from-scratch model.
- **Motion database:** collect sign videos, run `lexicon-add`, and agree on the shared format with A2S.
- **Visualization:** improve the avatar (3D or pix2pix) and add the face.
- **Evaluation:** design the human evaluation protocol with ASL stakeholders.
