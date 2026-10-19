import json

import cv2

from s2a.cli import main


def test_text_to_video_end_to_end(tmp_path, capsys):
    video = tmp_path / "out.mp4"
    main(["text-to-video", "--glosser", "rules", "--text", "My name is Ivy.", "--video", str(video),
          "--pose", str(tmp_path / "out.npz")])
    result = json.loads(capsys.readouterr().out)
    assert result["gloss"] == "POSS-1 NAME IVY"
    assert result["coverage"] == {"POSS-1": "missing", "NAME": "fingerspell", "IVY": "fingerspell"}
    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    assert (tmp_path / "out.npz").exists()


def test_lexicon_add_from_pose_file(tmp_path, capsys):
    from tests.test_gloss_to_pose import make_pose

    make_pose().save(tmp_path / "hello.npz")
    main(["lexicon-add", "--lexicon", str(tmp_path / "lex"), "--gloss", "hello", "--pose-file",
          str(tmp_path / "hello.npz")])
    assert "HELLO" in (tmp_path / "lex" / "index.csv").read_text()
