"""Build the gloss -> sign motion database from ASL videos.

Two sources:

* WLASL (2,000 glosses, ~21k videos; Computational Use of Data Agreement):
      s2a lexicon-wlasl --glosses hello name my library
      s2a lexicon-wlasl --top 100
* Any folder of videos, e.g. the team's own recordings:
      s2a lexicon-folder --videos recordings/            # HELLO.mp4 -> gloss HELLO
      s2a lexicon-folder --videos clips/ --labels labels.csv   # columns: file,gloss[,frame_start,frame_end]

Every video is converted to body + hand keypoints and then deleted, so no pixels, faces or
identities are kept: only motion. Glosses already in the lexicon are skipped, so a run that is
interrupted can simply be started again.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from s2a.gloss_to_pose.lexicon import DATA_LEXICON, ROOT, add_entry, normalize_gloss
from s2a.gloss_to_pose.pose import LEFT_HAND, NUM_BODY, RIGHT_HAND, SignPose

WLASL_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
WLASL_INDEX = ROOT / "data" / "WLASL_v0.3.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


@dataclass
class Clip:
    gloss: str
    url_or_path: str
    frame_start: int = 1
    frame_end: int = -1
    source: str = ""


# ---------- quality check ----------
def usable(pose: SignPose, min_hand_ratio: float = 0.2) -> bool:
    """A clip is usable if the body is tracked and a hand is visible in enough frames."""
    if len(pose) < 5:
        return False
    body = (pose.confidence[:, :NUM_BODY].max(axis=1) > 0).mean()
    hands = ((pose.confidence[:, LEFT_HAND].max(axis=1) > 0) | (pose.confidence[:, RIGHT_HAND].max(axis=1) > 0)).mean()
    return body >= 0.5 and hands >= min_hand_ratio


def existing_glosses(lexicon_dir: Path) -> set[str]:
    index = lexicon_dir / "index.csv"
    if not index.exists():
        return set()
    with open(index, newline="") as f:
        return {normalize_gloss(row["gloss"]) for row in csv.DictReader(f) if row["kind"] == "sign"}


# ---------- WLASL ----------
def load_wlasl(index_path: Path = WLASL_INDEX) -> list[dict]:
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the WLASL index to {index_path} ...")
        urllib.request.urlretrieve(WLASL_URL, index_path)
    return json.loads(index_path.read_text())


def wlasl_clips(entries: list[dict], glosses: list[str] | None, top: int) -> dict[str, list[Clip]]:
    """Candidate clips per gloss, in WLASL order (WLASL lists its most common glosses first)."""
    if glosses:
        wanted = [normalize_gloss(g) for g in glosses]
        by_gloss = {normalize_gloss(e["gloss"]): e for e in entries}
        missing = [g for g in wanted if g not in by_gloss]
        if missing:
            print(f"Not in WLASL (skipped): {' '.join(missing)}")
        entries = [by_gloss[g] for g in wanted if g in by_gloss]
    else:
        entries = entries[:top]

    clips = {}
    for entry in entries:
        gloss = normalize_gloss(entry["gloss"])
        candidates = []
        for inst in entry["instances"]:
            if inst["url"].lower().endswith(".swf"):  # Flash videos cannot be decoded
                continue
            candidates.append(Clip(gloss, inst["url"], inst.get("frame_start", 1), inst.get("frame_end", -1),
                                   f"WLASL video {inst['video_id']} (signer {inst.get('signer_id')}, "
                                   f"{inst.get('source')})"))
        clips[gloss] = candidates
    return clips


def download(url: str, folder: Path) -> Path:
    """Download one video. YouTube links need `pip install yt-dlp`."""
    if "youtube.com" in url or "youtu.be" in url:
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("YouTube video: install yt-dlp (`pip install yt-dlp`) to include these")
        out = folder / "video.%(ext)s"
        subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "mp4/best", "-o", str(out), url],
                       check=True, timeout=180, capture_output=True)
        return next(folder.glob("video.*"))
    path = folder / ("video" + (Path(url.split("?")[0]).suffix or ".mp4"))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)
    return path


# ---------- shared conversion loop ----------
def import_clips(clips: dict[str, list[Clip]], lexicon_dir: Path = DATA_LEXICON, per_gloss: int = 1,
                 max_tries: int = 6, skip_existing: bool = True) -> dict[str, int]:
    """For each gloss, convert candidate videos until `per_gloss` usable clips are saved."""
    from s2a.gloss_to_pose.extract import video_to_pose

    lexicon_dir = Path(lexicon_dir)
    have = existing_glosses(lexicon_dir) if skip_existing else set()
    saved = {}
    for n, (gloss, candidates) in enumerate(clips.items(), 1):
        prefix = f"[{n}/{len(clips)}] {gloss}:"
        if gloss in have:
            print(f"{prefix} already in the lexicon")
            continue
        count = 0
        for clip in candidates[:max_tries]:
            with tempfile.TemporaryDirectory() as tmp:  # the video is deleted as soon as this block ends
                try:
                    path = Path(clip.url_or_path)
                    if not path.exists():
                        path = download(clip.url_or_path, Path(tmp))
                    pose = video_to_pose(path, clip.frame_start, clip.frame_end)
                except Exception as e:  # dead links, unreadable files, ...
                    print(f"{prefix} skipped one video ({type(e).__name__}: {str(e)[:80]})")
                    continue
            if not usable(pose):
                print(f"{prefix} skipped one video (signer or hands not detected)")
                continue
            add_entry(lexicon_dir, pose, gloss, kind="sign", source=clip.source)
            count += 1
            if count >= per_gloss:
                break
        saved[gloss] = count
        print(f"{prefix} {'saved' if count else 'NO usable video found'}" + (f" ({count})" if count > 1 else ""))
    done = sum(1 for c in saved.values() if c)
    print(f"\nDone: {done}/{len(saved)} glosses added to {lexicon_dir}")
    return saved


# ---------- local folder ----------
def folder_clips(videos_dir: Path, labels: Path | None = None) -> dict[str, list[Clip]]:
    videos_dir = Path(videos_dir)
    clips: dict[str, list[Clip]] = {}
    if labels:
        with open(labels, newline="") as f:
            for row in csv.DictReader(f):
                gloss = normalize_gloss(row["gloss"])
                clips.setdefault(gloss, []).append(Clip(
                    gloss, str(videos_dir / row["file"]), int(row.get("frame_start") or 1),
                    int(row.get("frame_end") or -1), row.get("source") or f"local file {row['file']}"))
    else:
        for path in sorted(videos_dir.iterdir()):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                gloss = normalize_gloss(path.stem.split("__")[0])  # HELLO.mp4, HELLO__take2.mp4 -> HELLO
                clips.setdefault(gloss, []).append(Clip(gloss, str(path), source=f"local file {path.name}"))
    if not clips:
        raise ValueError(f"no videos found in {videos_dir}")
    return clips
