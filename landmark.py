import os
import json

import cv2
import dlib
import numpy as np
from tqdm import tqdm


LANDMARK_OFFSET_EYE_LEFT = 36
LANDMARK_OFFSET_EYE_RIGHT = 42
RECT = dlib.rectangle(869, 1508, 2019, 2658)


def shape_to_np(shape):
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    return shape_np


def shape_to_eyes(shape):
    eyes = np.zeros((12, 2), dtype="int")
    for i in range(0, 6):
        eyes[i] = (shape.part(LANDMARK_OFFSET_EYE_LEFT + i).x, shape.part(LANDMARK_OFFSET_EYE_LEFT + i).y)
        eyes[i + 6] = (shape.part(LANDMARK_OFFSET_EYE_RIGHT + i).x, shape.part(LANDMARK_OFFSET_EYE_RIGHT + i).y)
    return eyes


def shape_to_pupils(shape):
    left = np.zeros((6, 2), dtype="int")
    right = np.zeros((6, 2), dtype="int")
    for i in range(0, 6):
        left[i] = (shape.part(LANDMARK_OFFSET_EYE_LEFT + i).x, shape.part(LANDMARK_OFFSET_EYE_LEFT + i).y)
        right[i] = (shape.part(LANDMARK_OFFSET_EYE_RIGHT + i).x, shape.part(LANDMARK_OFFSET_EYE_RIGHT + i).y)
    pupils = np.array([np.mean(left, axis=0), np.mean(right, axis=0)], dtype="int")
    return pupils


def save_landmarks(path, landmarks):
    file_dir, _ = os.path.split(path)
    if not os.path.exists(file_dir):
        os.makedirs(path)
    with open(path, 'w') as f:
        json.dump(landmarks, f, indent=4)


def landmark(img_paths, path_shape_predictor, path_landmarks=None, dir_annotated=None):
    # Skip if landmark files already exists
    if os.path.isfile(path_landmarks):
        with open(path_landmarks) as f:
            return json.load(f)

    # Create directory for extracted photos
    if dir_annotated:
        if not os.path.exists(dir_annotated):
            os.makedirs(dir_annotated)

    # Prepare shape predictor
    predictor = dlib.shape_predictor(path_shape_predictor)

    # Run detection
    all_landmarks_pupils = dict()
    for img_file_path in tqdm(img_paths, desc='Detecting landmarks'):
        img = cv2.imread(img_file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = predictor(img_gray, RECT)
        landmarks_face = shape_to_np(shape)
        landmarks_pupils = shape_to_pupils(shape)

        # Draw stuff
        _, file_name = os.path.split(img_file_path)
        if dir_annotated:
            annotated_img_path = os.path.join(dir_annotated, file_name)
            if not os.path.isfile(annotated_img_path):
                landmark_eyes = shape_to_eyes(shape)
                for x, y in landmarks_face:
                    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
                for x, y in landmark_eyes:
                    cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                for x, y in landmarks_pupils:
                    cv2.circle(img, (x, y), 20, (255, 0, 0), -1)

                cv2.imwrite(annotated_img_path, img)
        all_landmarks_pupils[file_name] = landmarks_pupils.tolist()

    if path_landmarks:
        save_landmarks(path_landmarks, all_landmarks_pupils)

    return all_landmarks_pupils


if __name__ == '__main__':
    landmark('data/photos/resized/', 'data/shape_predictor_68_face_landmarks.dat')
