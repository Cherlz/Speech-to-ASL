"""Video -> pose extraction with MediaPipe Holistic (used to build the motion database).

Only body and hand keypoints are kept: no pixels and no face mesh, so the signer's face and
identity are not stored. Requires `pip install mediapipe`; the holistic model file is
downloaded on first use.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from s2a.gloss_to_pose.pose import LEFT_HAND, NUM_BODY, NUM_KEYPOINTS, RIGHT_HAND, SignPose

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
MODEL_PATH = Path.home() / ".cache" / "s2a" / "holistic_landmarker.task"


def _model_path() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe holistic model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def video_to_pose(video_path: str | Path, frame_start: int = 1, frame_end: int = -1) -> SignPose:
    """Extract poses from frames frame_start..frame_end (1-based, inclusive; -1 = until the end)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(_model_path())),
                                               running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frames, confs = [], []
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (frame_end > 0 and i + 1 > frame_end):
                break
            if i + 1 < frame_start:
                i += 1
                continue
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(i * 1000 / fps))
            data = np.zeros((NUM_KEYPOINTS, 3), np.float32)
            conf = np.zeros(NUM_KEYPOINTS, np.float32)
            for part, landmarks in ((slice(0, NUM_BODY), result.pose_landmarks),
                                    (LEFT_HAND, result.left_hand_landmarks),
                                    (RIGHT_HAND, result.right_hand_landmarks)):
                if landmarks:
                    # Pixel coordinates keep the aspect ratio correct for normalization.
                    data[part] = [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks]
                    conf[part] = 1.0
            frames.append(data)
            confs.append(conf)
            i += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames read from {video_path}")
    return SignPose(np.stack(frames), np.stack(confs), fps)
