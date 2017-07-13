import numpy as np
from sklearn.datasets.lfw import fetch_lfw_pairs
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces


def pad_img(img):
    return np.pad(img, ((0, 2), (0, 17)), mode='constant')


def get_data(subset):
    train_data = fetch_lfw_pairs(subset=subset)
    x1s, x2s = [], []
    for pair in train_data.pairs:
        x1s.append(pad_img(pair[0]))
        x2s.append(pad_img(pair[1]))
    return np.array(x1s), np.array(x2s), train_data.target
