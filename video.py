import os

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm


def read_img(path, device):
    return torchvision.io.read_image(path).to(device).permute(1, 2, 0)


def video(img_paths, dir_video, window_sizes=None):
    # Create directory for adjusted videos
    if not os.path.exists(dir_video):
        os.makedirs(dir_video)

    if window_sizes is None:
        window_sizes = [1]

    # Set computing device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_grad_enabled(False)
    height, width = cv2.imread(img_paths[0]).shape[:2]

    video_paths = []
    for win_size in window_sizes:
        # Prepare video writer
        path_video = os.path.join(dir_video, f'video_blend_{win_size:03d}.avi')
        vid = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'x264'), 60, (width, height))

        print(f'\nCreating video with window size {win_size:03d}')
        while True:
            print(f'Currently working with device type {device.type}')
            try:
                # Create dummy image and initialize image window
                dummy_image = torch.zeros_like(read_img(img_paths[0], device))
                window = dummy_image.unsqueeze(0)

                # Read initial images
                window_sum = torch.zeros_like(dummy_image, dtype=torch.int32)
                for img_path in tqdm(img_paths[:win_size - 1], desc=f'Loading initial images'):
                    next_img = read_img(img_path, device)
                    window_sum += next_img
                    window = torch.cat((window, next_img.unsqueeze(0)))

                # Iterate over rest of images and create video
                old_img_pointer = 0
                for img_path in tqdm(img_paths[win_size - 1:], desc=f'Creating video with window size {win_size:03d}'):
                    # Load new image
                    new_img = read_img(img_path, device)
                    old_img = window[old_img_pointer]
                    window_sum = window_sum + new_img - old_img
                    window[old_img_pointer] = new_img
                    old_img_pointer = (old_img_pointer + 1) % len(window)

                    # Convert and write to video
                    window_blend = window_sum / win_size
                    window_blend = window_blend.cpu().numpy().astype(np.uint8)
                    window_blend = cv2.cvtColor(window_blend, cv2.COLOR_RGB2BGR)
                    vid.write(window_blend)
                vid.release()
                video_paths.append(path_video)
            except RuntimeError:
                print('Device ran out of RAM, trying again with cpu')
                device = torch.device('cpu')
                continue
            break

    return video_paths


if __name__ == '__main__':
    pass
