import argparse
import pathlib
import os

from adjust import adjust
from extract import extract
from landmark import landmark
from video import video

SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'


def prepare_directories(working_dir, save_landmarks, create_annotated):
    working_dir = working_dir if working_dir else 'extracted'
    print(f'Working directory is set to "{working_dir}".')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    dir_resized = os.path.join(working_dir, 'resized')
    print(f'Extracted images will be saved into "{dir_resized}".')

    if save_landmarks:
        path_landmarks = os.path.join(working_dir, 'landmarks.json')
        print(f'Landmarks will be saved to "{path_landmarks}".')
    else:
        path_landmarks = None

    if create_annotated:
        dir_annotated = os.path.join(working_dir, 'annotated')
        print(f'Annotated images will be saved into "{dir_annotated}".')
    else:
        dir_annotated = None

    dir_adjusted = os.path.join(working_dir, 'adjusted')
    print(f'Adjusted images will be saved into "{dir_adjusted}".')

    dir_video = os.path.join(working_dir, 'videos')
    print(f'Videos will be saved into "{dir_video}".')

    return working_dir, dir_resized, path_landmarks, dir_annotated, dir_adjusted, dir_video


def main(db_path, photos_dir, working_dir, save_landmarks, create_annotated):
    working_dir, dir_resized, path_landmarks, dir_annotated, dir_adjusted, dir_video = prepare_directories(working_dir, save_landmarks, create_annotated)

    print()
    extracted = extract(db_path, photos_dir, dir_resized)
    landmarks = landmark(extracted, SHAPE_PREDICTOR, path_landmarks, dir_annotated)
    adjusted = adjust(extracted, landmarks, dir_adjusted)
    video(extracted, os.path.join(dir_video, 'unadjusted'), [1, 100])
    video(adjusted, dir_video, [1, 5, 10, 20, 50, 100, 300])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and process images from the Everyday app')
    parser.add_argument('db_path', type=pathlib.Path)
    parser.add_argument('photos_dir', type=pathlib.Path)
    parser.add_argument('-working_dir', '--wd', dest='working_dir', type=pathlib.Path, default=None)
    parser.add_argument('--save_landmarks', dest='save_landmarks', action='store_true', default=False)
    parser.add_argument('--create_annotated', dest='create_annotated', action='store_true', default=False)
    args = parser.parse_args()
    main(**vars(args))
