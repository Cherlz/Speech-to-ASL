import numpy as np
import pytest

from s2a.gloss_to_pose import gloss_to_pose
from s2a.gloss_to_pose.concatenate import active_span, concatenate
from s2a.gloss_to_pose.lexicon import Lexicon, add_entry, normalize_gloss
from s2a.gloss_to_pose.pose import (
    LEFT_SHOULDER, NUM_KEYPOINTS, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST, SignPose,
)


def make_pose(frames=20, raised=range(5, 15)) -> SignPose:
    data = np.zeros((frames, NUM_KEYPOINTS, 3), np.float32)
    data[:, LEFT_SHOULDER] = [300, 200, 0]
    data[:, RIGHT_SHOULDER] = [200, 200, 0]
    data[:, RIGHT_ELBOW] = [180, 300, 0]
    data[:, RIGHT_WRIST] = [180, 380, 0]  # hanging down
    for i in raised:
        data[i, RIGHT_WRIST] = [200, 250, 0]  # above the elbow
    return SignPose(data, np.ones((frames, NUM_KEYPOINTS)), 25.0)


def test_normalized_centers_and_scales_shoulders():
    pose = make_pose().normalized()
    np.testing.assert_allclose(pose.data[0, LEFT_SHOULDER, :2], [0.5, 0], atol=1e-6)
    np.testing.assert_allclose(pose.data[0, RIGHT_SHOULDER, :2], [-0.5, 0], atol=1e-6)


def test_resample_keeps_endpoints():
    pose = make_pose()
    short = pose.resample(7)
    assert len(short) == 7
    np.testing.assert_allclose(short.data[0], pose.data[0])
    np.testing.assert_allclose(short.data[-1], pose.data[-1])


def test_fill_gaps_bridges_short_dropouts_only():
    from s2a.gloss_to_pose.pose import RIGHT_HAND

    pose = make_pose(frames=30)
    pose.data[:, RIGHT_HAND, 0] = np.arange(30)[:, None]  # the hand moves steadily to the right
    pose.data[10:14, RIGHT_HAND] = 0
    pose.confidence[10:14, RIGHT_HAND] = 0  # 4-frame dropout: filled
    pose.confidence[20:, RIGHT_HAND] = 0  # lost until the end: stays empty
    filled = pose.fill_gaps(max_gap=8)
    assert (filled.confidence[10:14, RIGHT_HAND] > 0).all()
    np.testing.assert_allclose(filled.data[10:14, RIGHT_HAND, 0], np.arange(10, 14)[:, None] * np.ones((1, 21)))
    assert (filled.confidence[20:, RIGHT_HAND] == 0).all()
    assert (pose.fill_gaps(max_gap=3).confidence[10:14, RIGHT_HAND] == 0).all()  # gap too long


def test_active_span_finds_raised_hand():
    assert active_span(make_pose(), margin=0) == (5, 15)


def test_concatenate_adds_transitions_and_labels():
    out = concatenate([make_pose(), make_pose(), make_pose()], transition_seconds=0.2, labels=["A", "B", "C"])
    sign = 10 + 2 * 1  # raised frames 5-14 plus a 1-frame margin on each side
    rest = 4  # frames 0-3 before the hands rise, and 16-19 after they drop
    assert len(out) == (rest + sign) + 5 + sign + 5 + (sign + rest)
    assert out.labels.count("A") == rest + sign  # first sign keeps the hands rising from rest
    assert out.labels.count("B") == sign  # middle sign: hands stay up, no rest in between
    assert out.labels.count("C") == sign + rest  # last sign keeps the hands lowering
    assert out.labels[rest + sign:rest + sign + 5] == [""] * 5


def test_active_span_drops_raising_and_lowering():
    pose = make_pose(frames=20, raised=range(5, 15))
    pose.data[5, RIGHT_WRIST, 1] = pose.data[14, RIGHT_WRIST, 1] = 298  # just above the elbow: still on the way up / down
    assert active_span(pose, margin=0) == (6, 14)


def test_concatenate_caps_duration():
    out = concatenate([make_pose(frames=100, raised=range(0, 100))], max_seconds=[1.0])
    assert len(out) == 25


@pytest.fixture
def lexicon(tmp_path):
    for letter in "ABC":
        add_entry(tmp_path, make_pose(), letter, kind="letter")
    add_entry(tmp_path, make_pose(), "HELLO")
    add_entry(tmp_path, make_pose(), "IX-1")
    return Lexicon([tmp_path])


def test_lookup_sign_fingerspell_missing(lexicon):
    assert lexicon.lookup("hello").kind == "sign"
    m = lexicon.lookup("CAB")
    assert m.kind == "fingerspell" and len(m.clips) == 3
    assert lexicon.lookup("DOG").kind == "missing"
    assert lexicon.lookup("CAB", fingerspell=False).kind == "missing"


def test_pronouns_map_to_index_and_are_never_spelled(lexicon):
    assert normalize_gloss("X-I") == "IX-1"
    assert normalize_gloss("DESC-FAR") == "FAR"
    assert lexicon.lookup("X-I").kind == "sign"
    assert lexicon.lookup("POSS-1").kind == "missing"  # not fingerspelled as P-O-S-S
    assert normalize_gloss("X-MY") == "POSS-1"
    assert lexicon.lookup("X-MY").kind == "missing"  # not fingerspelled as M-Y
    assert normalize_gloss("X-Y") == "POSS-3PL"  # ASLG's mangled "their"


def test_name_captions_drop_the_x_tag(tmp_path):
    for letter in "ABC":
        add_entry(tmp_path, make_pose(), letter, kind="letter")
    pose, matches = gloss_to_pose("X-CAB", Lexicon([tmp_path]))
    assert matches[0].kind == "fingerspell"
    assert {label for label in pose.labels if label} == {"CAB (A)", "CAB (B)", "CAB (C)"}


def test_gloss_to_pose_skips_punctuation(lexicon):
    pose, matches = gloss_to_pose("HELLO CAB .", lexicon)
    assert [m.kind for m in matches] == ["sign", "fingerspell"]
    assert len(pose) > 0 and pose.labels[0] == "HELLO"


def test_save_load_roundtrip(tmp_path):
    pose = make_pose()
    pose.save(tmp_path / "p.npz")
    loaded = SignPose.load(tmp_path / "p.npz")
    np.testing.assert_allclose(loaded.data, pose.data, rtol=1e-3)
    assert loaded.fps == pose.fps


def test_bundled_fingerspelling_lexicon_covers_alphabet():
    lexicon = Lexicon()
    assert sorted(lexicon.letters) == [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def test_transition_keeps_a_hand_seen_at_only_one_end():
    from s2a.gloss_to_pose.concatenate import transition
    from s2a.gloss_to_pose.pose import RIGHT_HAND

    a, b = make_pose(), make_pose()
    a.data[:, RIGHT_HAND] = 1.0
    b.confidence[0, RIGHT_HAND] = 0  # tracker missed the hand on b's first frame
    between = transition(a, b, 5)
    assert (between.confidence[:, RIGHT_HAND] > 0).all()
    np.testing.assert_allclose(between.data[:, RIGHT_HAND], 1.0)  # held where it was last seen
