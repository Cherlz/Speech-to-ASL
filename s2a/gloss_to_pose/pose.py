"""A minimal pose-sequence container.

We keep 75 MediaPipe Holistic keypoints per frame:
    0-32   body (POSE_LANDMARKS)
    33-53  left hand
    54-74  right hand
Each keypoint has (x, y, z) plus a confidence; confidence 0 means "not detected".
Poses are stored on disk as compressed .npz files so the whole motion database
is plain numpy and easy for both the S2A and A2S teams to share.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_BODY = 33
NUM_HAND = 21
LEFT_HAND = slice(NUM_BODY, NUM_BODY + NUM_HAND)
RIGHT_HAND = slice(NUM_BODY + NUM_HAND, NUM_BODY + 2 * NUM_HAND)
NUM_KEYPOINTS = NUM_BODY + 2 * NUM_HAND

# MediaPipe body indices we rely on.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24


@dataclass
class SignPose:
    data: np.ndarray  # (frames, 75, 3) float32
    confidence: np.ndarray  # (frames, 75) float32
    fps: float = 25.0
    labels: list[str] | None = None  # optional per-frame caption (e.g. the gloss being signed)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        if self.data.ndim != 3 or self.data.shape[1:] != (NUM_KEYPOINTS, 3):
            raise ValueError(f"expected data of shape (frames, {NUM_KEYPOINTS}, 3), got {self.data.shape}")
        if self.confidence.shape != self.data.shape[:2]:
            raise ValueError("confidence must have shape (frames, keypoints)")

    def __len__(self) -> int:
        return len(self.data)

    @property
    def seconds(self) -> float:
        return len(self) / self.fps

    def slice(self, start: int, end: int) -> "SignPose":
        return SignPose(self.data[start:end], self.confidence[start:end], self.fps)

    # ---------- I/O ----------
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, data=self.data.astype(np.float16), confidence=self.confidence.astype(np.float16),
                            fps=np.float32(self.fps))

    @classmethod
    def load(cls, path: str | Path) -> "SignPose":
        path = Path(path)
        if path.suffix == ".pose":
            return cls.from_pose_format(path)
        with np.load(path) as f:
            return cls(f["data"], f["confidence"], float(f["fps"]))

    @classmethod
    def from_pose_format(cls, path: str | Path) -> "SignPose":
        """Read a MediaPipe Holistic `.pose` file (sign-language-processing/pose-format)."""
        from pose_format import Pose

        with open(path, "rb") as f:
            pose = Pose.read(f.read())
        header = pose.header

        def component(name: str) -> slice:
            c = next(c for c in header.components if c.name == name)
            start = header.get_point_index(name, c.points[0])
            return slice(start, start + len(c.points))

        parts = [component("POSE_LANDMARKS"), component("LEFT_HAND_LANDMARKS"), component("RIGHT_HAND_LANDMARKS")]
        data = np.ma.getdata(pose.body.data)[:, 0]
        conf = pose.body.confidence[:, 0]
        return cls(np.concatenate([data[:, p] for p in parts], axis=1),
                   np.concatenate([conf[:, p] for p in parts], axis=1), float(pose.body.fps))

    # ---------- geometry ----------
    def normalized(self) -> "SignPose":
        """Center on the shoulder midpoint and scale so shoulder width == 1.

        Different signers / cameras then share one coordinate system, which is what lets
        us stitch signs from different videos together (and map them onto one avatar).
        """
        conf = self.confidence
        both = (conf[:, LEFT_SHOULDER] > 0) & (conf[:, RIGHT_SHOULDER] > 0)
        if not both.any():
            return self
        ls, rs = self.data[both, LEFT_SHOULDER], self.data[both, RIGHT_SHOULDER]
        center = ((ls + rs) / 2).mean(axis=0)
        width = np.linalg.norm((ls - rs)[:, :2], axis=1).mean()
        if width <= 0:
            return self
        return SignPose((self.data - center) / width, conf.copy(), self.fps)

    def resample(self, num_frames: int) -> "SignPose":
        """Linearly time-resample to num_frames, keeping fps (i.e. speeds the sign up or down)."""
        if num_frames == len(self) or len(self) < 2:
            return self
        old = np.linspace(0, 1, len(self))
        new = np.linspace(0, 1, num_frames)
        idx = np.clip(np.searchsorted(old, new, side="right") - 1, 0, len(self) - 2)
        t = ((new - old[idx]) / (old[idx + 1] - old[idx]))[:, None, None]
        data = self.data[idx] * (1 - t) + self.data[idx + 1] * t
        conf = np.minimum(self.confidence[idx], self.confidence[idx + 1])
        return SignPose(data, conf, self.fps)


def empty_pose(fps: float = 25.0) -> SignPose:
    return SignPose(np.zeros((0, NUM_KEYPOINTS, 3)), np.zeros((0, NUM_KEYPOINTS)), fps)
