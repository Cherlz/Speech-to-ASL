"""Stitch individual sign clips into one continuous signing sequence."""
from __future__ import annotations

import numpy as np

from s2a.gloss_to_pose.pose import (
    LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST, SignPose, empty_pose,
)


def active_span(pose: SignPose, margin: int = 3) -> tuple[int, int]:
    """[start, end) frames where a hand is raised above its elbow, i.e. the sign itself.

    Dictionary clips start and end with the hands at rest; dropping that makes the
    stitched sentence look far more like fluent signing.
    """
    raised = np.zeros(len(pose), dtype=bool)
    for wrist, elbow in ((LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)):
        seen = (pose.confidence[:, wrist] > 0) & (pose.confidence[:, elbow] > 0)
        raised |= seen & (pose.data[:, wrist, 1] < pose.data[:, elbow, 1])  # image y grows downwards
    if not raised.any():
        return 0, len(pose)
    first = int(np.argmax(raised))
    last = len(pose) - int(np.argmax(raised[::-1]))
    return max(0, first - margin), min(len(pose), last + margin)


def transition(a: SignPose, b: SignPose, frames: int) -> SignPose:
    """Linear interpolation from a's last frame to b's first frame (movement epenthesis)."""
    if frames <= 0 or len(a) == 0 or len(b) == 0:
        return empty_pose(a.fps)
    t = np.linspace(0, 1, frames + 2)[1:-1, None, None]
    data = a.data[-1] * (1 - t) + b.data[0] * t
    conf = np.repeat(np.minimum(a.confidence[-1], b.confidence[0])[None], frames, axis=0)
    return SignPose(data, conf, a.fps)


def concatenate(poses: list[SignPose], max_seconds: list[float] | None = None, transition_seconds: float = 0.2,
                trim: bool = True, fps: float = 25.0, labels: list[str] | None = None) -> SignPose:
    """Normalize, trim, speed-cap and join sign clips with smooth transitions.

    max_seconds[i] caps the duration of sign i (citation-form dictionary signs are much
    slower than fluent signing; fingerspelled letters should be faster still).
    labels[i], if given, captions every frame of sign i in the result.
    """
    if not poses:
        return empty_pose(fps)
    max_seconds = max_seconds or [None] * len(poses)
    labels = labels or [""] * len(poses)
    clips, clip_labels = [], []
    for pose, cap, label in zip(poses, max_seconds, labels):
        pose = pose.normalized().fill_gaps()
        if trim:
            pose = pose.slice(*active_span(pose))
        if pose.fps != fps:
            pose = SignPose(pose.data, pose.confidence, fps).resample(max(1, round(len(pose) * fps / pose.fps)))
        if cap is not None and pose.seconds > cap:
            pose = pose.resample(max(2, round(cap * fps)))
        if len(pose):
            clips.append(pose)
            clip_labels.append(label)
    if not clips:
        return empty_pose(fps)

    gap = round(transition_seconds * fps)
    parts, frame_labels = [clips[0]], [clip_labels[0]] * len(clips[0])
    for prev, nxt, label in zip(clips, clips[1:], clip_labels[1:]):
        between = transition(prev, nxt, gap)
        parts += [between, nxt]
        frame_labels += [""] * len(between) + [label] * len(nxt)
    return SignPose(np.concatenate([p.data for p in parts]), np.concatenate([p.confidence for p in parts]), fps,
                    labels=frame_labels)
