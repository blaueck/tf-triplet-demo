import os
import glob
import gzip

import numpy as np
import tensorflow as tf


def _glob(pattern):

    if isinstance(pattern, str):
        files = glob.glob(pattern)
    elif isinstance(pattern, list):
        files = []
        for p in pattern:
            files.extend(glob.glob(pattern))
    else:
        raise TypeError('wrong argument type.')

    return files


def get_cifar10(files):

    images_splits = []
    labels_splits = []
    n_pixel = 32 * 32 * 3

    for f in files:
        buffer = np.fromfile(f, dtype='uint8')
        buffer = buffer.reshape(-1, n_pixel+1)

        images = buffer[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = buffer[:, 0]

        images_splits.append(images)
        labels_splits.append(labels)

    images = np.concatenate(images_splits)
    labels = np.concatenate(labels_splits)

    return images, labels


def get_cifar100(files):

    images_splits = []
    labels_splits = []
    n_pixel = 32 * 32 * 3

    for f in files:
        buffer = np.fromfile(f, dtype='uint8')
        buffer = buffer.reshape(-1, n_pixel+2)

        images = buffer[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = buffer[:, 1]

        images_splits.append(images)
        labels_splits.append(labels)

    images = np.concatenate(images_splits)
    labels = np.concatenate(labels_splits)

    return images, labels


def get_mnist(image_files, label_files):

    images_splits = []
    labels_splits = []

    for i_f, l_f in zip(image_files, label_files):

        with gzip.open(i_f, 'rb') as f:
            images = np.frombuffer(f.read(), dtype='uint8', offset=16)
            images = images.reshape(-1, 28, 28, 1)
            images_splits.append(images)

        with gzip.open(l_f, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype='uint8', offset=8)
            labels_splits.append(labels)

    images = np.concatenate(images_splits)
    labels = np.concatenate(labels_splits)

    return images, labels


def load_data(root, dataset, is_training):

    if dataset == 'cifar10':

        if is_training:
            pattern = os.path.join(root, 'data_batch*.bin')
        else:
            pattern = os.path.join(root, 'test_batch.bin')

        files = _glob(pattern)
        assert files, 'no file is matched.'

        data = get_cifar10(files)
        meta = {'n_class': 10}

    elif dataset == 'cifar100':

        if is_training:
            pattern = os.path.join(root, 'train.bin')
        else:
            pattern = os.path.join(root, 'test.bin')

        files = _glob(pattern)
        assert files, 'no file is matched.'

        data = get_cifar100(pattern)
        meta = {'n_class': 100}

    elif dataset == 'mnist':

        if is_training:
            img_pattern, label_pattern = [
                os.path.join(root, fn)
                for fn in ['train-images-idx3-ubyte.gz',
                           'train-labels-idx1-ubyte.gz']]
        else:
            img_pattern, label_pattern = [
                os.path.join(root, fn)
                for fn in ['t10k-images-idx3-ubyte.gz',
                           't10k-labels-idx1-ubyte.gz']]
        
        img_files, label_files = _glob(img_pattern), _glob(label_pattern)
        assert img_files, 'no image file is matched.'
        assert label_files, 'no label file is matched.'

        data = get_mnist(img_files, label_files)
        meta = {'n_class': 10}

    else:
        raise ValueError('%s is not supported.' % dataset)
    
    meta['name'] = dataset
    return data, meta


def split_data(data, rate, shuffle=True):

    images, labels = data
    N = images.shape[0]
    split_point = int(N * rate)

    if shuffle:
        idx = np.random.permutation(N)
        images, labels = images[idx], labels[idx]

    train_data = images[split_point:], labels[split_point:]
    val_data = images[:split_point], labels[:split_point]

    return train_data, val_data


def build_data_map(images, labels, n_class):
    data_map = {}
    for i in range(n_class):
        data_map[i] = images[labels == i].copy()
    return data_map


class DataSampler:

    def __init__(self, data, n_class, n_class_per_iter, n_img_per_class):
        images, labels = data
        self.data_map = build_data_map(images, labels, n_class)
        self.n_classs = n_class
        self.n_class_per_iter = n_class_per_iter
        self.n_img_per_class = n_img_per_class

    def __iter__(self):
        return self

    def __next__(self):
        choiced_classes = np.random.choice(self.n_classs, self.n_class_per_iter, replace=False)
        images = []
        labels = []
        for c in choiced_classes:
            cimgs = self.data_map[c]
            choiced_imgs = np.random.choice(cimgs.shape[0], self.n_img_per_class, replace=False)
            images.append(cimgs[choiced_imgs])
            clabels = np.empty([self.n_img_per_class], dtype='int32')
            clabels[:] = c
            labels.append(clabels)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        return images, labels


def preprocess_for_train(image, label):
    shape = image.get_shape().as_list()
    image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
    image = tf.random_crop(image, shape)
    # image = tf.image.random_flip_left_right(image)
    image = (tf.to_float(image) - 127.5) / 128.

    label = tf.to_int64(label)
    return image, label


def preprocess_for_eval(image, label):
    image = (tf.to_float(image) - 127.5) / 128.

    label = tf.to_int64(label)
    return image, label