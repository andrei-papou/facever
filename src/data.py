import os
import numpy as np
from scipy.ndimage import rotate, shift
from sklearn.datasets.lfw import fetch_lfw_pairs
from config import DATA_DIRECTORY
from utils import unison_shuffle


ROTATIONS_RANGE = range(1, 9)
SHIFTS_RANGE = range(1, 5)
IMG_SLICE = (slice(0, 64), slice(0, 64))


def pad_img(img):
    return np.pad(img, ((0, 2), (0, 17)), mode='constant')


def get_data(subset, data_directory=DATA_DIRECTORY):
    train_data = fetch_lfw_pairs(subset=subset)

    x1s_path, x2s_path, ys_path = (os.path.join(data_directory, '{}.npy'.format(nm)) for nm in ('x1s', 'x2s', 'ys'))

    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)
    else:
        x1s = np.load(x1s_path)
        x2s = np.load(x2s_path)
        ys = np.load(ys_path)
        return x1s, x2s, ys

    x1s_pre, x2s_pre, ys_pre = [], [], train_data.target
    for (img1, img2) in train_data.pairs:
        x1s_pre.append(img1)
        x2s_pre.append(img2)

    x1s_pre, x2s_pre, ys_pre = unison_shuffle([x1s_pre, x2s_pre, ys_pre], len(ys_pre))

    x1s, x2s, ys = [], [], []
    for (i1, i2, label) in zip(x1s_pre, x2s_pre, ys_pre):
        img1 = pad_img(i1)
        img2 = pad_img(i2)

        # original images
        x1s.append(img1)
        x2s.append(img2)
        ys.append(label)

        # rotated images
        for angle in ROTATIONS_RANGE:
            x1s.append(np.asarray(rotate(img1, angle))[IMG_SLICE])
            x2s.append(np.asarray(rotate(img2, angle))[IMG_SLICE])
            ys.append(label)
            x1s.append(np.asarray(rotate(img1, -angle))[IMG_SLICE])
            x2s.append(np.asarray(rotate(img2, -angle))[IMG_SLICE])
            ys.append(label)

        # shifted images
        for pixels_to_shift in SHIFTS_RANGE:
            x1s.append(shift(img1, pixels_to_shift))
            x2s.append(shift(img2, pixels_to_shift))
            ys.append(label)
            x1s.append(shift(img1, -pixels_to_shift))
            x2s.append(shift(img2, -pixels_to_shift))
            ys.append(label)

    x1s, x2s, ys = (np.array(x) for x in (x1s, x2s, ys))
    np.save(x1s_path, x1s)
    np.save(x2s_path, x2s)
    np.save(ys_path, ys)

    return x1s, x2s, ys
