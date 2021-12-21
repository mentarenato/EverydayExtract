import os
import json
import math

import cv2
import dlib
import numpy as np
from tqdm import tqdm


LANDMARK_OFFSET_EYE_LEFT = 36
LANDMARK_OFFSET_EYE_RIGHT = 42
RECT = dlib.rectangle(869, 1508, 2019, 2658)
PATH_PHOTOS = '../data/photos/resized/'


def calculate_target_pupil_coords(coordinates_pupils):
    left = [coords[0] for coords in coordinates_pupils]
    right = [coords[1] for coords in coordinates_pupils]
    left = np.mean(left, axis=0, dtype=float)
    right = np.mean(right, axis=0, dtype=float)
    left[1] = right[1] = np.mean([left[1], right[1]])
    return left, right


def get_translation_matrix(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy]
    ])


def adjust(img_paths, pupil_positions, dir_adjusted):
    # Make sure there is landmark data for each image
    if len(img_paths) != len(pupil_positions):
        error_msg = 'Number of images does not match number of data points. '
        error_msg += f'{len(img_paths)} images and {len(pupil_positions)} data points were provided.'
        raise ValueError(error_msg)

    # Create directory for adjusted photos
    if not os.path.exists(dir_adjusted):
        os.makedirs(dir_adjusted)

    # Load pupil coordinates
    pupil_positions = sorted([(k, v) for k, v in pupil_positions.items()])
    pupil_positions = np.array([coords for _, coords in pupil_positions])
    pupil_target_left, pupil_target_right = calculate_target_pupil_coords(pupil_positions)
    pupil_target_distance = np.linalg.norm(pupil_target_right - pupil_target_left)

    # Load images and adjust
    adjusted = []
    for img_path, (pupil_left, pupil_right) in tqdm(zip(img_paths, pupil_positions), desc='Adjusting images', total=len(img_paths)):
        # Create file name for adjusted image, skip if already exists
        _, img_name = os.path.split(img_path)
        adjusted_img_path = os.path.join(dir_adjusted, img_name)
        adjusted.append(adjusted_img_path)
        if os.path.isfile(adjusted_img_path):
            continue

        # Load image
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Calc scale
        pupil_distance = np.linalg.norm(pupil_right - pupil_left)
        scale = pupil_target_distance / pupil_distance

        # Calc angle
        dy, dx = pupil_right[1] - pupil_left[1], pupil_right[0] - pupil_left[0]
        angle = math.atan2(dy, dx) * 180 / math.pi

        # Calc translation
        dx, dy = pupil_left - pupil_target_left

        # Rotate/resize image
        rot_matrix = cv2.getRotationMatrix2D(pupil_left.tolist(), angle, scale)
        trans_matrix = get_translation_matrix(-dx, -dy)
        img = cv2.warpAffine(img, rot_matrix, (width, height))
        img = cv2.warpAffine(img, trans_matrix, (width, height))

        # Write image to disk
        cv2.imwrite(adjusted_img_path, img)
    return adjusted
