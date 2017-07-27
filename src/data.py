import random
import numpy as np
from scipy.ndimage import rotate, shift, zoom
from sklearn.datasets.lfw import fetch_lfw_pairs, fetch_lfw_people
from utils import unison_shuffle


class DataAugmenter:

    def __init__(self, params_list, func, img_slice):
        self.params_list = params_list
        self.func = func
        self.img_slice = img_slice

    def modify(self, img):
        param = random.choice(self.params_list)
        return self.func(img, param)[self.img_slice]


ROTATIONS_RANGE = range(1, 25)
SHIFTS_RANGE = range(1, 18)
ZOOM_RANGE = (1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4)
IMG_SLICE = (slice(0, 64), slice(0, 64))
TEST_TO_TRAIN_LIMIT = 900


def pad_img(img):
    return np.pad(img, ((0, 2), (0, 17)), mode='constant')


def get_data(multiplier: int):
    train_data = fetch_lfw_pairs(subset='train')
    test_data = fetch_lfw_pairs(subset='test')
    test_pairs, test_labels = unison_shuffle((test_data.pairs, test_data.target), len(test_data.target))

    x1s_trn, x2s_trn, ys_trn, x1s_vld, x2s_vld = [], [], [], [], []

    augmenters = [
        DataAugmenter(params_list=ROTATIONS_RANGE, func=rotate, img_slice=IMG_SLICE),
        DataAugmenter(params_list=SHIFTS_RANGE, func=shift, img_slice=IMG_SLICE),
        DataAugmenter(params_list=ZOOM_RANGE, func=zoom, img_slice=IMG_SLICE)
    ]

    for (pair, y) in zip(train_data.pairs, train_data.target):
        img1, img2 = pad_img(pair[0]), pad_img(pair[1])
        x1s_trn.append(img1)
        x2s_trn.append(img2)
        ys_trn.append(y)

        if multiplier:
            for i in range(multiplier):
                modifier1 = random.choice(augmenters)
                modifier2 = random.choice(augmenters)
                x1s_trn.append(modifier1.modify(img1))
                x2s_trn.append(modifier2.modify(img2))
                ys_trn.append(y)

    for (img1, img2) in test_pairs:
        x1s_vld.append(pad_img(img1))
        x2s_vld.append(pad_img(img2))

    return (
        np.array(x1s_trn),
        np.array(x2s_trn),
        np.array(ys_trn),
        np.array(x1s_vld),
        np.array(x2s_vld),
        np.array(test_labels)
    )


def compose_pairs(images, labels):
    x1s, x2s, ys = [], [], []
    for img1, y1 in zip(images, labels):
        for img2, y2 in zip(images, labels):
            x1s.append(pad_img(img1))
            x2s.append(pad_img(img2))
            ys.append(int(y1 == y2))
    return np.array(x1s), np.array(x2s), np.array(ys)
