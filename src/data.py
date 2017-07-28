from random import randint
import numpy as np
from scipy.ndimage import rotate, shift, zoom
from sklearn.datasets.lfw import fetch_lfw_pairs, fetch_lfw_people
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from constants import ROTATIONS_RANGE, SHIFTS_RANGE, ZOOM_RANGE, IMG_SLICE
from utils import unison_shuffle


def pad_img(img):
    return np.pad(img, ((0, 2), (0, 17)), mode='constant')


def get_data(rotations=False, shifts=False, zooming=False):
    train_data = fetch_lfw_pairs(subset='train')
    test_data = fetch_lfw_pairs(subset='test')

    x1s_trn, x2s_trn, ys_trn, x1s_vld, x2s_vld = [], [], [], [], []

    for pair, label in zip(train_data.pairs, train_data.target):
        img1 = pad_img(pair[0])
        img2 = pad_img(pair[1])

        x1s_trn.append(img1)
        x2s_trn.append(img2)
        ys_trn.append(label)

        if rotations:
            for angle in ROTATIONS_RANGE:
                x1s_trn.append(np.asarray(rotate(img1, angle=angle))[IMG_SLICE])
                x2s_trn.append(np.asarray(rotate(img2, angle=angle))[IMG_SLICE])
                ys_trn.append(label)
        if shifts:
            for slide in SHIFTS_RANGE:
                x1s_trn.append(np.asarray(shift(img1, slide))[IMG_SLICE])
                x2s_trn.append(np.asarray(shift(img2, slide))[IMG_SLICE])
                ys_trn.append(label)
        if zooming:
            for scale in ZOOM_RANGE:
                x1s_trn.append(np.asarray(zoom(img1, scale))[IMG_SLICE])
                x2s_trn.append(np.asarray(zoom(img2, scale))[IMG_SLICE])
                ys_trn.append(label)

    for (img1, img2) in test_data.pairs:
        x1s_vld.append(pad_img(img1))
        x2s_vld.append(pad_img(img2))

    return (
        np.array(x1s_trn),
        np.array(x2s_trn),
        np.array(ys_trn),
        np.array(x1s_vld),
        np.array(x2s_vld),
        np.array(test_data.target)
    )


class DataProvider:

    def __init__(self, batch_size: int, training_test_boundary: int = 13000):
        self.batch_size = batch_size
        self.images_train, self.labels_train = None, None
        self.images_test, self.labels_test = None, None
        self.x1s_lfw_pairs, self.x2s_lfw_pairs, self.ys_lfw_pairs = [], [], []
        self.images_olv, self.labels_olv = [], []
        self.training_test_boundary = training_test_boundary

    def fetch_data(self):
        dataset = fetch_lfw_people()
        self.images_train = [pad_img(img) for img in dataset.images[:self.training_test_boundary]]
        self.images_test = [pad_img(img) for img in dataset.images[self.training_test_boundary:13200]]
        self.labels_train = list(dataset.target[:self.training_test_boundary])
        self.labels_test = list(dataset.target[self.training_test_boundary:13200])

        pairs_dataset = fetch_lfw_pairs(subset='test')
        for pair, label in zip(pairs_dataset.pairs, pairs_dataset.target):
            self.x1s_lfw_pairs.append(pad_img(pair[0]))
            self.x2s_lfw_pairs.append(pad_img(pair[1]))
            self.ys_lfw_pairs.append(label)

        olv_dataset = fetch_olivetti_faces()
        self.images_olv, self.labels_olv = olv_dataset.images, olv_dataset.target

    @property
    def batches(self):
        data_size = len(self.labels_train)
        images, labels = unison_shuffle((self.images_train, self.labels_train), data_size)
        x1s, x2s, ys = [], [], []
        non_eq_x1, non_eq_x2 = None, None
        for i in range(data_size):
            for j in range(data_size):
                if i == j:
                    continue
                img1 = images[i]
                img2 = images[j]
                y1 = labels[i]
                y2 = labels[j]
                if int(y1 == y2):
                    x1s.append(img1)
                    x2s.append(img2)
                    ys.append(1)
                    if non_eq_x1 is not None and non_eq_x2 is not None:
                        x1s.append(non_eq_x1)
                        x2s.append(non_eq_x2)
                        ys.append(0)
                else:
                    non_eq_x1 = img1
                    non_eq_x2 = img2
                if len(ys) >= self.batch_size:
                    yield x1s, x2s, ys
                    x1s, x2s, ys = [], [], []

    @property
    def evaluation_data(self):
        data_size = len(self.labels_test)
        images, labels = unison_shuffle((self.images_test, self.labels_test), data_size)
        x1s, x2s, ys = [], [], []
        non_eq_x1, non_eq_x2 = None, None
        for i in range(data_size):
            for j in range(data_size):
                if i == j:
                    continue
                img1 = images[i]
                img2 = images[j]
                y1 = labels[i]
                y2 = labels[j]
                if int(y1 == y2):
                    x1s.append(img1)
                    x2s.append(img2)
                    ys.append(1)
                    if non_eq_x1 is not None and non_eq_x2 is not None:
                        x1s.append(non_eq_x1)
                        x2s.append(non_eq_x2)
                        ys.append(0)
        while len(ys) % self.batch_size:
            i = randint(0, len(ys) - 1)
            x1s.pop(i)
            x2s.pop(i)
            ys.pop(i)
        print('batch of size {} used for evaluation.'.format(sum(ys) * 2))
        return x1s, x2s, ys

    @property
    def evaluation_data_lfw_pairs(self):
        return self.x1s_lfw_pairs, self.x2s_lfw_pairs, self.ys_lfw_pairs

    @property
    def evaluation_data_olivetti(self):
        data_size = len(self.labels_olv)
        images, labels = unison_shuffle((self.images_olv, self.labels_olv), data_size)
        x1s, x2s, ys = [], [], []
        non_eq_x1, non_eq_x2 = None, None
        for i in range(data_size):
            for j in range(data_size):
                if i == j:
                    continue
                img1 = images[i]
                img2 = images[j]
                y1 = labels[i]
                y2 = labels[j]
                if int(y1 == y2):
                    x1s.append(img1)
                    x2s.append(img2)
                    ys.append(1)
                    if non_eq_x1 is not None and non_eq_x2 is not None:
                        x1s.append(non_eq_x1)
                        x2s.append(non_eq_x2)
                        ys.append(0)
        while len(ys) % self.batch_size:
            i = randint(0, len(ys) - 1)
            x1s.pop(i)
            x2s.pop(i)
            ys.pop(i)
        print('batch of size {} used for Olivetti evaluation.'.format(sum(ys) * 2))
        return x1s, x2s, ys
