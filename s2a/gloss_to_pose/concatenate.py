"""Stitch individual sign clips into one continuous signing sequence."""
from __future__ import annotations

import numpy as np

from s2a.gloss_to_pose.pose import (
    LEFT_ELBOW, LEFT_HAND, LEFT_WRIST, NUM_BODY, RIGHT_ELBOW, RIGHT_HAND, RIGHT_WRIST, SignPose, empty_pose,
)

REST_SECONDS = 0.4  # how long the sentence starts and ends with hands at rest


def active_span(pose: SignPose, margin: int = 1, settle: float = 0.35) -> tuple[int, int]:
    """[start, end) frames of the sign itself, without raising the hands from rest or lowering them.

    A hand counts as signing while its wrist is above its elbow *and* has risen at least
    (1 - settle) of the way from its resting height to its typical signing height.
    """
    active = np.zeros(len(pose), dtype=bool)
    for wrist, elbow in ((LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)):
        seen = (pose.confidence[:, wrist] > 0) & (pose.confidence[:, elbow] > 0)
        y = pose.data[:, wrist, 1]  # image y grows downwards
        raised = seen & (y < pose.data[:, elbow, 1])
        if not raised.any():
            continue
        signing_y = np.median(y[raised])
        rest_y = np.median(y[seen & ~raised]) if (seen & ~raised).any() else y[raised].max()
        active |= raised & (y <= signing_y + settle * (rest_y - signing_y))
    if not active.any():
        return 0, len(pose)
    first = int(np.argmax(active))
    last = len(pose) - int(np.argmax(active[::-1]))
    return max(0, first - margin), min(len(pose), last + margin)


def transition(a: SignPose, b: SignPose, frames: int) -> SignPose:
    """Linear interpolation from a's last frame to b's first frame (movement epenthesis).

    A body part seen at only one end is held at that end's position, so a hand that the tracker
    missed for a frame does not vanish during the transition.
    """
    if frames <= 0 or len(a) == 0 or len(b) == 0:
        return empty_pose(a.fps)
    start, end = a.data[-1].copy(), b.data[0].copy()
    start_conf, end_conf = a.confidence[-1], b.confidence[0]
    for part in (slice(0, NUM_BODY), LEFT_HAND, RIGHT_HAND):
        seen_a, seen_b = start_conf[part].max() > 0, end_conf[part].max() > 0
        if seen_a and not seen_b:
            end[part] = start[part]
        elif seen_b and not seen_a:
            start[part] = end[part]
    t = np.linspace(0, 1, frames + 2)[1:-1, None, None]
    data = start * (1 - t) + end * t
    conf = np.repeat(np.maximum(start_conf, end_conf)[None], frames, axis=0)
    return SignPose(data, conf, a.fps)


def concatenate(poses: list[SignPose], max_seconds: list[float] | None = None, transition_seconds: float = 0.2,
                trim: bool = True, fps: float = 25.0, labels: list[str] | None = None) -> SignPose:
    """Normalize, trim, speed-cap and join sign clips with smooth transitions.

    max_seconds[i] caps the duration of sign i (citation-form dictionary signs are much
    slower than fluent signing; fingerspelled letters should be faster still).
    labels[i], if given, captions every frame of sign i in the result.
    The first sign keeps its hands-rising onset and the last sign its hands-lowering offset, at
    normal speed, so the sentence starts and ends at rest; signs in between go hand to hand.
    """
    if not poses:
        return empty_pose(fps)
    max_seconds = max_seconds or [None] * len(poses)
    labels = labels or [""] * len(poses)
    clips, clip_labels = [], []
    last = len(poses) - 1
    for i, (pose, cap, label) in enumerate(zip(poses, max_seconds, labels)):
        pose = pose.normalized().fill_gaps()
        if pose.fps != fps:
            pose = SignPose(pose.data, pose.confidence, fps).resample(max(1, round(len(pose) * fps / pose.fps)))
        start, end = active_span(pose) if trim else (0, len(pose))
        sign = pose.slice(start, end)
        if cap is not None and sign.seconds > cap:
            sign = sign.resample(max(2, round(cap * fps)))
        rest = round(REST_SECONDS * fps)
        parts = [pose.slice(max(0, start - rest), start)] if trim and i == 0 else []  # hands rising from rest
        parts += [sign]
        parts += [pose.slice(end, min(len(pose), end + rest))] if trim and i == last else []  # hands lowering
        pose = SignPose(np.concatenate([p.data for p in parts]), np.concatenate([p.confidence for p in parts]), fps)
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
