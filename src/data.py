import os
from random import randint
import numpy as np
from sklearn.datasets.lfw import fetch_lfw_pairs, fetch_lfw_people
from sklearn.datasets.olivetti_faces import fetch_olivetti_faces
from scipy.misc import imread, imresize
from utils import unison_shuffle
from constants import IMG_SLICE


def pad_img(img):
    return np.pad(img, ((0, 2), (0, 17)), mode='constant')


def load_own_images(directory, img_size):
    images = []
    labels = []
    label = 1
    container_dir = os.path.expanduser(directory)
    for label_directory in os.listdir(container_dir):
        for file_name in os.listdir(label_directory):
            img = np.array(imread(os.path.join(container_dir, label_directory, file_name), flatten=True))
            size_scalar = img_size / min(img.shape[0], img.shape[1])
            img = imresize(img, size_scalar)[IMG_SLICE:IMG_SLICE]
            images.append(img)
            labels.append(label)
        label += 1
    return images, labels


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

    def _build_pairs(self, images: np.array, labels: np.array, purpose: str):
        data_size = len(labels)
        images, labels = unison_shuffle((images, labels), data_size)
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
        print('batch of size {} used for {}.'.format(sum(ys) * 2, purpose))
        return x1s, x2s, ys

    @property
    def evaluation_data(self):
        return self._build_pairs(self.images_test, self.labels_test, purpose='LFW evaluation')

    @property
    def evaluation_data_lfw_pairs(self):
        return self.x1s_lfw_pairs, self.x2s_lfw_pairs, self.ys_lfw_pairs

    @property
    def evaluation_data_olivetti(self):
        return self._build_pairs(self.images_olv, self.labels_olv, purpose='Olivetti evaluation')
