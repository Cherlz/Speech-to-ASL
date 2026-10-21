# S2A: Speech/Text → ASL (low-fidelity prototype)

Type or say an English sentence. The program turns it into **ASL gloss** using **our own AI model**, then shows a stick figure signing it.

![The S2A demo page](docs/demo.png)

- [Part 1: Run the demo (step by step, no experience needed)](#part-1-run-the-demo)
- [Part 2: How it works](#part-2-how-it-works)
- [Part 3: Results and data sources](#part-3-results-and-data-sources)
- [Part 4: For developers](#part-4-for-developers)
- [Limitations](#limitations-please-read-before-the-design-review) · [Next steps for the team](#next-steps-for-the-s2a-team)

---

# Part 1: Run the demo

You **don't** need to train anything. The trained model is already included. These steps are for a **Mac**. Windows users should see the [Windows notes](#windows-notes).

> ⚠️ **Paste ONE command at a time.** Press **Enter**, and wait until you see the `%` prompt again before pasting the next one. If you paste several lines together, Terminal can treat them as answers to the previous command and things break.

## First time only (about 10 minutes)

### Step 1: Check you have Python
Open **Terminal**: press **⌘ Command + Space**, type `Terminal`, then press **Enter**. Paste:
```bash
python3 --version
```
- If it shows **Python 3.10 or higher**, you're good.
- If it shows 3.9 or `command not found`, install Python from https://www.python.org/downloads/, then close and reopen Terminal.
- If a popup asks to install "command line developer tools", click **Install**.

### Step 2: Make sure you have ~3 GB of free space
Go to Apple menu  → **System Settings** → **General** → **Storage**. The AI library (PyTorch) is big. If you're low on space, empty the Trash and delete old downloads first.

### Step 3: Download the project
1. Open https://github.com/Cherlz/Speech-to-ASL. You must be logged in to GitHub with access to the repo.
2. Click the green **`<> Code`** button → **Download ZIP**.
3. Open **Downloads** in Finder and **double-click** `Speech-to-ASL-main.zip`. A folder called `Speech-to-ASL-main` appears.

> Why not `git clone`? The repo is private, so `git clone` asks for a GitHub login, and normal passwords don't work there. The ZIP avoids that. If you already use GitHub Desktop or a GitHub token, cloning works too.

### Step 4: Go into the project folder
```bash
cd ~/Downloads/Speech-to-ASL-main
```

### Step 5: Create the project's private Python space (a "virtual environment")
```bash
python3 -m venv .venv
```
This makes a hidden `.venv` folder that keeps the project's tools separate from the rest of your Mac.

### Step 6: Activate it (step inside the space)
```bash
source .venv/bin/activate
```
✅ Your Terminal line should now start with **`(.venv)`**. If it doesn't, don't continue. Go back to Step 4.

### Step 7: Install everything (2–5 minutes)
```bash
pip install -e .
```
✅ Wait until the last lines say **`Successfully installed ...`**.

### Step 8: Start the demo
```bash
s2a demo
```
✅ Terminal says `S2A demo running at http://localhost:8000` and your browser opens the demo page.

## Using the demo
- Type a sentence and press **Translate**, or press **Enter**.
- Click **🎤 Speak** to say a sentence out loud. This works best in **Chrome**. Click **Allow** when it asks for the microphone.
- Click one of the **"Try:"** examples.
- Use the **Glosser** dropdown to compare **our model** with the simple **rule baseline**.
- The colored chips show how each word was handled: 🟢 **sign** (real sign from our dictionary), 🟠 **fingerspelled** (spelled letter by letter), 🔴 **missing**.

> **Keep Terminal open while you use the demo.** Terminal is the "engine". If you close it, the page stops working.

## Stopping and exiting
1. **Stop the demo:** click on the Terminal window and press **Control + C**. Use the `control` key, not ⌘. It prints `stopped`.
2. **Leave the virtual environment** (optional):
   ```bash
   deactivate
   ```
   The `(.venv)` disappears from the start of the line.
3. **Quit Terminal:** **⌘ Command + Q**. If it asks *"terminate running processes?"*, click **Terminate**. That only means the demo was still running.

Nothing is lost. Everything stays in the `Speech-to-ASL-main` folder.

## Every time after that (3 commands)
Open Terminal and paste one at a time:
```bash
cd ~/Downloads/Speech-to-ASL-main
```
```bash
source .venv/bin/activate
```
```bash
s2a demo
```

## Other things you can try (with `(.venv)` active)
Translate English into gloss only:
```bash
s2a text-to-gloss --text "Where is the library?"
```
Save a signing animation as a GIF and open it:
```bash
s2a text-to-video --text "My name is Cheryl." --video cheryl.gif
```
```bash
open cheryl.gif
```
The text in quotes after `--text` is what gets translated. The name after `--video` is the file it creates and must end in `.gif` or `.mp4`.

## Getting the latest version
Your downloaded folder does **not** update by itself when the GitHub repo changes. To update:
1. Quit Terminal. Delete the old `Speech-to-ASL-main` folder and `.zip` from Downloads, then **Empty Trash**.
2. Repeat **Steps 3–8**. It's faster the second time because the big downloads are cached.

## Troubleshooting

| What you see | What it means and what to do |
|---|---|
| `No space left on device` | Your disk is full. Free about 3 GB (Step 2), run `python3 -m pip cache purge`, then delete the broken setup with `rm -rf .venv` and redo Steps 5–7. |
| `Username for 'https://github.com'` | You used `git clone` on the private repo. Press **Control + C** and use the ZIP download (Step 3) instead. |
| `command not found: s2a` or `command not found: python` | The virtual environment isn't active, or the install didn't finish. Run Step 6, check for `(.venv)`, then Step 7. |
| `no such file or directory: .venv/bin/activate` | Step 5 didn't run, or you're in the wrong folder. Run Step 4, then Step 5. |
| `does not appear to be a Python project` | You're in the wrong folder. Run Step 4 first. |
| `Address already in use` | The demo is already running in another Terminal window. Use that one, or run `s2a demo --port 8001`. |
| Browser didn't open | Open Chrome and go to **http://localhost:8000** yourself. |
| 🎤 button greyed out or not working | Use Chrome, and allow microphone access (the lock icon in the address bar → Microphone → Allow). |
| `[notice] A new release of pip is available` | Harmless. Ignore it. |

## Windows notes
Use **Command Prompt** instead of Terminal, with the same steps except:
- Step 1: `py --version`
- Step 4: `cd %USERPROFILE%\Downloads\Speech-to-ASL-main`
- Step 5: `py -m venv .venv`
- Step 6: `.venv\Scripts\activate`
- Stop the demo with **Ctrl + C**.

---

# Part 2: How it works

The pipeline is a relay race with four steps:

```
🎤 "Where is the library?"
   │  1. Speech → text   (browser speech recognition, or OpenAI Whisper on the command line)
📝 Where is the library?
   │  2. Text → gloss    ← OUR OWN AI MODEL (a Transformer we trained from scratch)
🔤 WHERE BE LIBRARY ?
   │  3. Gloss → poses   (look up each word in the lexicon; fingerspell if it's not there)
🦴 body + hand keypoints, frame by frame
   │  4. Poses → video   (draw a stick figure)
🎬 animation
```

**Key words**
- **Gloss:** a written form of ASL that uses English words in CAPITALS, following ASL grammar. It's the middle step between English and signing.
- **Model:** the AI. It learned English → gloss from about 82,000 example pairs. That learning is called **training**.
- **Pose:** the position of the body and fingers in one video frame, stored as 75 points (33 body points and 21 for each hand).
- **Lexicon:** our sign dictionary, linking a gloss word to a pose recording of that sign. Right now it has **only the alphabet A–Z**, so every word is fingerspelled. Adding whole-word signs is the most useful next step.

This project follows the design of [sign-language-processing/spoken-to-signed-translation](https://github.com/sign-language-processing/spoken-to-signed-translation). The main difference is step 2: they use hand-written rules or external services, and **we train our own neural network**.

| Step | Code | Ours or reused |
|---|---|---|
| Speech → text | `s2a/speech_to_text.py`, demo page | Reused (Whisper / browser) |
| Text → gloss | `s2a/text_to_gloss/` | **Ours**: tokenizer and Transformer encoder-decoder, trained from scratch in PyTorch |
| Gloss → pose | `s2a/gloss_to_pose/` | Ours: lexicon lookup, fingerspelling fallback, smooth stitching |
| Video → pose (for adding signs) | `s2a/gloss_to_pose/extract.py` | Google MediaPipe |
| Pose → video | `s2a/pose_to_video.py` | Ours (stick figure) |
| Demo page | `s2a/demo.py` | Ours (Python server, HTML/CSS/JavaScript page) |

**Languages:** Python for everything, plus HTML/CSS/JavaScript for the demo page. Main libraries: PyTorch, Hugging Face `tokenizers`, NumPy, OpenCV and sacrebleu.

---

# Part 3: Results and data sources

## How good is the model?
Scores on the 1,000-sentence ASLG-PC12 **test set**, which the model never saw during training:

| | Rule baseline | **Our model** |
|---|---|---|
| BLEU (higher is better, max 100) | 22.8 | **94.3** |
| Word error rate (lower is better) | 47.9% | **2.8%** |
| Sentences exactly right | 2.2% | **81.2%** |

The model has 7.6M parameters and was trained for 5 epochs, which took about 40 minutes on a laptop-class CPU. Training history is in `models/text2gloss/history.json`.

⚠️ These scores measure agreement with ASLG-PC12's gloss, **not** whether a Deaf signer would find the output correct. See [Limitations](#limitations-please-read-before-the-design-review).

## Where the data comes from
1. **Training sentences: ASLG-PC12** (Othman & Jemni, 2012). About 87,000 English/ASL-gloss pairs built from European Parliament text. We use the train/dev/test split published with Yin & Read (2020), [kayoyin/transformer-slt](https://github.com/kayoyin/transformer-slt). It is downloaded by `scripts/prepare_aslg.py` and not stored in this repo.
2. **Fingerspelling A–Z poses:** from [sign.mt](https://sign.mt), via the MIT-licensed spoken-to-signed-translation project. They were converted to our format in `assets/fingerspelling/`, and the license is in that folder.

---

# Part 4: For developers

## Install options
```bash
pip install -e ".[dev]"       # + pytest
pip install -e ".[speech]"    # + Whisper, for `s2a speech-to-video` (also needs ffmpeg)
pip install -e ".[extract]"   # + MediaPipe, for `s2a lexicon-add --video`
```

## Train or evaluate the text → gloss model
```bash
python scripts/prepare_aslg.py                     # download ASLG-PC12 → data/aslg/
python -m s2a.text_to_gloss.train --data data/aslg --out models/text2gloss --epochs 5 --warmup 800
python -m s2a.text_to_gloss.evaluate --data data/aslg --split test --model models/text2gloss
python -m s2a.text_to_gloss.evaluate --data data/aslg --split test --glosser rules   # baseline
```
- The architecture has 3 encoder and 3 decoder layers, d_model 256 and 4 heads, with a shared 8k BPE vocabulary and tied embeddings. It uses label smoothing, AdamW and a warmup + inverse-sqrt learning-rate schedule.
- `--device auto` uses a GPU (CUDA or Apple MPS) when one is available.
- To train on a better dataset, write it as `{train,dev,test}.en` and `{train,dev,test}.gloss`, one sentence per line, aligned line by line.
- ⚠️ Training to the same `--out` folder **overwrites** the included model.

## Command reference
```bash
s2a demo [--port 8000] [--lexicon DIR ...]
s2a text-to-gloss  --text "..." [--glosser model|rules] [--beam 5]
s2a text-to-video  --text "..." --video out.gif [--pose out.npz] [--lexicon DIR ...] [--no-fingerspell]
s2a speech-to-video --audio clip.wav --video out.gif
s2a lexicon-add --lexicon DIR --gloss WORD (--video sign.mp4 | --pose-file sign.pose) [--source "..."]
```

## Growing the lexicon (shared motion database with A2S)
A lexicon is a folder of `.npz` pose clips plus an `index.csv` (`path,gloss,kind,source`):
```bash
s2a lexicon-add --lexicon data/lexicon --gloss HELLO --video hello.mp4 --source "signer name / dataset"
s2a demo --lexicon data/lexicon
```
- Each clip is `(frames, 75, 3)` keypoints plus per-point confidence, which A2S can reuse for recognition.
- How gloss is matched to clips (`s2a/gloss_to_pose/lexicon.py`):
  - Markers are stripped, so `DESC-FAR` is looked up as `FAR`.
  - ASLG pronouns map to pointing, possessive and reflexive signs, so `X-I` becomes `IX-1`, `X-MY` becomes `POSS-1` and `X-MYSELF` becomes `SELF-1`. Name captions drop the tag, so `X-IVY` is shown as `IVY`.
  - `IX-*`, `POSS-*` and `SELF-*` are never fingerspelled.
- Candidate sign sources: WLASL, ASL Citizen (check their licenses), or recordings of fluent signers made with their consent.

## Tests
```bash
pytest -q
```
The tests cover pose maths, lexicon lookup and fingerspelling, a model overfitting a toy dataset (checks training and decoding), and the end-to-end CLI.

## Project layout
```
s2a/
  cli.py                 s2a command
  demo.py                browser demo
  speech_to_text.py      Whisper wrapper
  text_to_gloss/         tokenizer, model, data, train, evaluate, rule baseline
  gloss_to_pose/         pose format, lexicon, stitching, MediaPipe extraction
  pose_to_video.py       stick-figure renderer
models/text2gloss/       trained model + tokenizer
assets/fingerspelling/   A–Z pose clips
scripts/prepare_aslg.py  dataset download
tests/
```

---

# Limitations (please read before the design review)
1. **ASLG-PC12 is not real ASL.** Its gloss was generated by rules from English, not written by signers. That's why outputs keep English words like `BE` and `TO`. We need a dataset checked by ASL experts, which is the stakeholder outreach from the 10/12 meeting.
2. **The model copies ASLG's conventions:** `X-` pronouns, `DESC-` adjectives, and mostly English word order.
3. **No facial expressions.** ASL grammar uses the face (questions, negation, etc.), and our poses only include the body and hands.
4. **Stitched dictionary signs are not fluent signing.** There's no role shift, use of space or classifiers. This is a feasibility demo.
5. **The lexicon only has A–Z**, so every word is currently fingerspelled.
6. **Automatic scores are only a proxy.** We still need a human evaluation with fluent signers.

# Next steps for the S2A team
- **Data:** find or build a better English↔gloss dataset with ASL experts, and agree on a gloss convention (IX-1/IX-2/IX-3, etc.).
- **Model:** try other architectures and settings, and compare against fine-tuning a pretrained model such as T5.
- **Lexicon:** collect whole-word sign videos and add them with `s2a lexicon-add`. Agree on the shared format with A2S.
- **Visualization:** add the face, and a better avatar (3D or photoreal).
- **Evaluation:** design a human evaluation with ASL stakeholders.
