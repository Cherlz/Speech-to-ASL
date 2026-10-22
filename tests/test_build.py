import json

import numpy as np
import pytest

from s2a.cli import main
from s2a.gloss_to_pose import build
from s2a.gloss_to_pose.lexicon import Lexicon
from s2a.gloss_to_pose.pose import LEFT_HAND, RIGHT_HAND
from tests.test_gloss_to_pose import make_pose

WLASL = [
    {"gloss": "hello", "instances": [
        {"url": "https://example.com/old.swf", "video_id": "1", "signer_id": 1, "source": "aslpro"},
        {"url": "https://example.com/hello.mp4", "video_id": "2", "signer_id": 5, "source": "x",
         "frame_start": 3, "frame_end": 40}]},
    {"gloss": "thank you", "instances": [{"url": "https://example.com/ty.mp4", "video_id": "3"}]},
    {"gloss": "my", "instances": [{"url": "https://example.com/my.mp4", "video_id": "4"}]},
]


@pytest.fixture
def fake_extract(monkeypatch):
    """Replace MediaPipe with a stub that records which videos and frame ranges were requested."""
    calls = []

    def video_to_pose(path, frame_start=1, frame_end=-1):
        calls.append((str(path), frame_start, frame_end))
        if "broken" in str(path):
            raise ValueError("cannot decode")
        pose = make_pose()
        if "nohands" in str(path):
            pose.confidence[:, LEFT_HAND] = 0
            pose.confidence[:, RIGHT_HAND] = 0
        return pose

    monkeypatch.setattr("s2a.gloss_to_pose.extract.video_to_pose", video_to_pose)
    monkeypatch.setattr(build, "download", lambda url, folder: (folder / "v.mp4").write_bytes(b"") or folder / "v.mp4")
    return calls


def test_wlasl_clips_selects_glosses_and_skips_flash():
    clips = build.wlasl_clips(WLASL, ["Hello", "thank you", "unicorn"], top=100)
    assert list(clips) == ["HELLO", "THANK-YOU"]
    assert [c.url_or_path for c in clips["HELLO"]] == ["https://example.com/hello.mp4"]
    assert (clips["HELLO"][0].frame_start, clips["HELLO"][0].frame_end) == (3, 40)
    assert list(build.wlasl_clips(WLASL, None, top=2)) == ["HELLO", "THANK-YOU"]


def test_import_wlasl_builds_lexicon_and_resumes(tmp_path, fake_extract):
    clips = build.wlasl_clips(WLASL, None, top=3)
    saved = build.import_clips(clips, tmp_path)
    assert saved == {"HELLO": 1, "THANK-YOU": 1, "MY": 1}
    assert fake_extract[0][1:] == (3, 40)  # WLASL frame range is passed through
    lexicon = Lexicon([tmp_path])
    assert lexicon.lookup("thank you").kind == "sign"
    assert lexicon.lookup("X-MY").kind == "sign"  # POSS-1 falls back to the MY sign
    assert "signer 5" in (tmp_path / "index.csv").read_text()

    fake_extract.clear()
    assert build.import_clips(clips, tmp_path) == {}  # second run skips what is already there
    assert fake_extract == []


def test_import_falls_back_to_next_video(tmp_path, fake_extract):
    clips = {"HELLO": [build.Clip("HELLO", str(tmp_path / "broken.mp4")),
                       build.Clip("HELLO", str(tmp_path / "nohands.mp4")),
                       build.Clip("HELLO", str(tmp_path / "good.mp4"))]}
    for name in ("broken.mp4", "nohands.mp4", "good.mp4"):
        (tmp_path / name).write_bytes(b"")
    assert build.import_clips(clips, tmp_path / "lex") == {"HELLO": 1}
    assert [c[0].rsplit("/", 1)[1] for c in fake_extract] == ["broken.mp4", "nohands.mp4", "good.mp4"]
    assert len(Lexicon([tmp_path / "lex"]).signs) == 1


def test_folder_clips_from_filenames_and_csv(tmp_path):
    for name in ("HELLO.mp4", "HELLO__take2.mov", "thank you.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    clips = build.folder_clips(tmp_path)
    assert {g: len(c) for g, c in clips.items()} == {"HELLO": 2, "THANK-YOU": 1}

    (tmp_path / "labels.csv").write_text("file,gloss,frame_start,frame_end\nHELLO.mp4,greet,10,50\n")
    clips = build.folder_clips(tmp_path, tmp_path / "labels.csv")
    assert list(clips) == ["GREET"] and (clips["GREET"][0].frame_start, clips["GREET"][0].frame_end) == (10, 50)


def test_usable_requires_body_and_hands():
    pose = make_pose()
    assert build.usable(pose)
    pose.confidence[:, LEFT_HAND] = 0
    pose.confidence[:, RIGHT_HAND] = 0
    assert not build.usable(pose)


def test_gloss_to_video_cli(tmp_path, capsys):
    lex = tmp_path / "lex"
    make_pose().save(tmp_path / "hello.npz")
    main(["lexicon-add", "--lexicon", str(lex), "--gloss", "hello", "--pose-file", str(tmp_path / "hello.npz")])
    capsys.readouterr()
    main(["gloss-to-video", "--gloss", "HELLO IX-1 CAB", "--lexicon", str(lex), "--video", str(tmp_path / "o.gif")])
    result = json.loads(capsys.readouterr().out)
    assert result["coverage"] == {"HELLO": "sign", "IX-1": "missing", "CAB": "fingerspell"}
    assert (tmp_path / "o.gif").stat().st_size > 0
    assert np.isfinite(np.load(tmp_path / "hello.npz")["data"].astype(float)).all()
