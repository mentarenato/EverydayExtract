import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


PATH_PHOTOS = '../data/photos/resized/'


def main():
    img_file_paths = sorted([os.path.join(PATH_PHOTOS, file_name) for file_name in os.listdir(PATH_PHOTOS)])
    imgs = []
    for img_file_path in tqdm(img_file_paths, desc='Reading images'):
        imgs.append(np.array(Image.open(img_file_path)))

    img_avg = np.average(imgs)
    plt.imshow(img_avg / 255, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()
