import json
import os
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerResult
from PIL import Image


def get_image_paths():
    paths = (Path(os.getcwd()) / "data/letters_v2").iterdir()
    return paths


def resize(image_path: Path, new_width=192, new_height=192):
    image = Image.open(image_path)

    image.thumbnail((new_width, new_height))
    if hasattr(image, "filename"):
        path = Path(image.filename)
        save_dir = path.parent.parent / "letters_resized"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving {path.name} to {save_dir}")
        image.save(save_dir / f"{path.stem}_cropped{path.suffix}")


def build_detector(model_name="hand_landmarker.task"):
    model_path = Path(os.getcwd()) / "models"
    base_options = python.BaseOptions(model_asset_path=str(model_path / model_name))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


def save_keypoints(image: Path, result: HandLandmarkerResult):
    save_dir = image.parent.parent / "keypoints_v2"
    os.makedirs(save_dir, exist_ok=True)

    keypoints = []
    all_x = np.array([landmark.x for landmark in result.hand_landmarks[0]])
    all_y = np.array([landmark.y for landmark in result.hand_landmarks[0]])
    all_z = np.array([landmark.z for landmark in result.hand_landmarks[0]])

    for landmark in result.hand_landmarks[0]:
        keypoints.append(
            {
                "x": (landmark.x - all_x.min()) / (all_x.max() - all_x.min()),
                "y": (landmark.y - all_y.min()) / (all_y.max() - all_y.min()),
                "z": (landmark.z - all_z.min()) / (all_z.max() - all_z.min()),
                "presence": landmark.presence,
                "visibility": landmark.visibility,
            }
        )

        with open(f"{save_dir}/{image.stem}.json", "w", encoding="utf-8") as f:
            json.dump(keypoints, f, ensure_ascii=False, indent=4)


def main():
    image_paths = get_image_paths()
    detector = build_detector()

    for index, image_path in enumerate(image_paths):
        # TODO: resize to 192 x 192?
        # resize(image_path)
        image = mp.Image.create_from_file(str(image_path))
        result = detector.detect(image)
        save_keypoints(image_path, result)


if __name__ == "__main__":
    main()
