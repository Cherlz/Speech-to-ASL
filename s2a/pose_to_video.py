"""Render a pose sequence as a stick-figure video (the low-fidelity "avatar")."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from s2a.gloss_to_pose.pose import LEFT_HAND, RIGHT_HAND, SignPose

# Upper-body MediaPipe connections (legs are irrelevant for signing).
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (9, 10)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

BODY_COLOR = (200, 200, 200)
LEFT_COLOR = (80, 160, 255)  # BGR
RIGHT_COLOR = (255, 140, 60)


def _edges():
    yield from ((a, b, BODY_COLOR, 3) for a, b in BODY_EDGES)
    for offset, color in ((LEFT_HAND.start, LEFT_COLOR), (RIGHT_HAND.start, RIGHT_COLOR)):
        yield from ((offset + a, offset + b, color, 2) for a, b in HAND_EDGES)


def render_frames(pose: SignPose, size: int = 512, captions: list[str] | None = None) -> list[np.ndarray]:
    import cv2

    captions = captions or pose.labels

    # Poses are normalized (shoulder width 1, centered on shoulders): place shoulders a bit above center.
    scale, cx, cy = size / 3.2, size / 2, size * 0.42
    frames = []
    for i in range(len(pose)):
        img = np.full((size, size, 3), 30, np.uint8)
        pts = pose.data[i, :, :2] * scale + (cx, cy)
        seen = pose.confidence[i] > 0
        for a, b, color, width in _edges():
            if seen[a] and seen[b]:
                cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, width, cv2.LINE_AA)
        if seen[0]:  # head, plus eyes so the facing direction is visible
            r = int(0.35 * scale)
            cv2.circle(img, tuple(int(v) for v in pts[0]), r, BODY_COLOR, 2, cv2.LINE_AA)
            for eye in (2, 5):
                if seen[eye]:
                    cv2.circle(img, tuple(int(v) for v in pts[eye]), 3, BODY_COLOR, -1, cv2.LINE_AA)
        if captions and captions[i]:
            cv2.putText(img, captions[i], (16, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
        frames.append(img)
    return frames


def write_video(pose: SignPose, path: str | Path, size: int = 512, captions: list[str] | None = None) -> Path:
    """Write .mp4 (OpenCV) or .gif (Pillow) depending on the file extension."""
    import cv2

    path = Path(path)
    frames = render_frames(pose, size, captions)
    if path.suffix.lower() == ".gif":
        from PIL import Image

        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / pose.fps), loop=0)
    else:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (size, size))
        for f in frames:
            writer.write(f)
        writer.release()
    return path

