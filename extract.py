import os
import sqlite3
from datetime import datetime

import matplotlib as plt
from PIL import Image
from tqdm import tqdm


TIMESTAMP_OFFSET = 978307200


class Key:
    TIMESTAMP = 'timestamp'
    FILEPATH = 'filepath'
    DIMENSIONS = 'dimensions'


def secs_to_days(secs):
    return int(secs / (60 * 60 * 24))


def fetch_db_data(path_db):
    # Establish DB connection and read table
    con = sqlite3.connect(path_db)
    cur = con.cursor()
    data = cur.execute('SELECT ZCREATED, ZIDENTIFIER FROM ZPHOTO').fetchall()
    return [dict(zip((Key.TIMESTAMP, Key.FILEPATH), item)) for item in sorted(data)]


def plot_data(data):
    min_timestamp = data[0][Key.TIMESTAMP]
    days_passed = secs_to_days(data[-1][Key.TIMESTAMP] - min_timestamp)
    plt.plot([item[Key.TIMESTAMP] for item in data])
    plt.show()
    plt.figure(figsize=(37, 1))
    plt.hist([secs_to_days(item[Key.TIMESTAMP] - min_timestamp) for item in data], bins=days_passed+1)
    plt.show()
    plt.hist([int(datetime.fromtimestamp(item[Key.TIMESTAMP]).strftime('%H')) for item in data], bins=24)
    plt.show()


def extract(path_db, dir_photos, dir_extracted, target_dim=None):
    # Create directory for extracted photos
    if not os.path.exists(dir_extracted):
        os.makedirs(dir_extracted)

    # Adjust timestamp and target file path
    data = fetch_db_data(path_db)
    data = [dict(zip(
        (Key.TIMESTAMP, Key.FILEPATH),
        (item[Key.TIMESTAMP] + TIMESTAMP_OFFSET, os.path.join(dir_photos, item[Key.FILEPATH] + '.jpg'))
    )) for item in data]

    # If no target dimension is specified, load all images and use max dimension
    if not target_dim:
        dims = set()
        for item in tqdm(data, desc='Reading images'):
            if not os.path.isfile(item[Key.FILEPATH]):
                continue
            with Image.open(item[Key.FILEPATH]) as img:
                item[Key.DIMENSIONS] = img.width, img.height
                dims.add((img.width, img.height))
        target_dim = max(dims)

    # Resize images
    extracted = []
    for item in tqdm(data, desc='Resizing images'):
        if not os.path.isfile(item[Key.FILEPATH]):
            continue

        img_date = datetime.fromtimestamp(item[Key.TIMESTAMP]).strftime('%y-%m-%d_%H-%m')
        _, src_file_name = os.path.split(item[Key.FILEPATH])

        target_file_name = img_date + src_file_name
        target_file_path = os.path.join(dir_extracted, target_file_name)
        extracted.append(target_file_path)
        if os.path.isfile(target_file_path):
            continue

        img = Image.open(item[Key.FILEPATH])
        img = img.resize(target_dim)
        img.save(target_file_path)
        img.close()

    return extracted


if __name__ == '__main__':
    PATH_DB = '../data/Everyday2.sqlite'
    PATH_PHOTOS = '../data/photos'
    extract(PATH_DB, PATH_PHOTOS)
