import numpy as np
import platform
import os.path as path

from tensorflow.examples.tutorials.mnist import input_data

from . import ROOT_PATH

PATH_DATA = path.join(ROOT_PATH, 'mnist_data')
PATH_MNIST_INDICES = path.join(PATH_DATA, 'mnist_digits.npy')

def get_mnist(N=50000, Ns=5000, onehot=False, digit=1):
    mnist = input_data.read_data_sets(PATH_DATA, one_hot=False)
    im_data_full, lab_data_full = mnist.train.images, mnist.train.labels

    train_ind = (lab_data_full == digit)
    test_ind = (lab_data_full == digit)

    im_train = im_data_full[train_ind, :]
    lab_train = lab_data_full[train_ind]

    im_test = im_data_full[test_ind, :]
    lab_test = lab_data_full[test_ind]

    Y = im_train.astype(float)  # outputs are the images
    Ys = im_test.astype(float)
    X = lab_train.astype(int)[:, None]  # inputs are the labels
    Xs = lab_test.astype(int)[:, None]

    return X[:N, :], Y[:N, :], Xs[:Ns, :], Ys[:Ns, :]

def get_mnist_all_classes(N_class=10, Ns_class=10):
    mnist = input_data.read_data_sets(PATH_DATA, one_hot=False)
    im_data_full, lab_data_full = mnist.train.images, mnist.train.labels
    im_data_test, lab_data_test = mnist.test.images, mnist.test.labels

    train_idx, test_idx = [], []
    for digit in range(10):
        train_ind = np.argwhere(lab_data_full == digit).flatten()
        test_ind = np.argwhere(lab_data_test == digit).flatten()
        np.random.seed(0)
        np.random.shuffle(train_ind)
        np.random.shuffle(test_ind)
        train_idx.append(train_ind[:N_class])
        test_idx.append(test_ind[:Ns_class])

    train_idx = np.array(train_idx).flatten()
    test_idx = np.array(test_idx).flatten()

    im_train = im_data_full[train_idx, :]
    lab_train = lab_data_full[train_idx]
    im_test = im_data_test[test_idx, :]
    lab_test = lab_data_test[test_idx]

    Y = im_train.astype(float)  # outputs are the images
    Ys = im_test.astype(float)
    X = lab_train.astype(int)[:, None]  # inputs are the labels
    Xs = lab_test.astype(int)[:, None]

    return X, Y, Xs, Ys

import matplotlib.pyplot as plt
INDICES = np.load(PATH_MNIST_INDICES)
MNIST = input_data.read_data_sets(PATH_DATA, one_hot=True)

def get_mnist_full_test():
    """
    return the full mnist test set
    """
    im_data_test, lab_data_test = MNIST.test.images, MNIST.test.labels
    im_test = im_data_test.astype(np.float32)
    lab_test = lab_data_test.astype(np.float32)

    return im_test, lab_test


def get_mnist_cvae(Nc):
    """
    Returns mnist with (Nc) ** 2 images per class, for all classes (0..9)
    """
    im_data_full, lab_data_full = MNIST.train.images, MNIST.train.labels
    im_train = im_data_full[INDICES[Nc-1], :].astype(np.float32)
    lab_train = lab_data_full[INDICES[Nc-1]].astype(np.float32)

    shuffled_idx = np.arange(len(im_train))
    np.random.shuffle(shuffled_idx)

    return im_train[shuffled_idx], lab_train[shuffled_idx]


if __name__ == "__main__":
    # idx = np.load("mnist_digits.npy")
    # mnist = input_data.read_data_sets(PATH, one_hot=False)
    # im_data_full, lab_data_full = mnist.train.images, mnist.train.labels

    # d = 2

    # im_train = im_data_full[idx[d-1], :]
    # lab_train = lab_data_full[idx[d-1]]
    Nc = 2
    X, Y = get_mnist_cvae(Nc)
    print(X.shape)
    print(Y.shape)

    fig, axes = plt.subplots(Nc ** 2, 10)
    for i, ax in enumerate(axes.flatten("F")):
        ax.imshow(X[i, ...].reshape(28, 28))
        print(Y[i, ...])
    plt.show()


    # indices = []
    # for e in range(1, 13):
    #     num = 2 ** e
    #     idx = get_mnist_all_classes(num, 0)
    #     indices.append(idx)
    # np.save("mnist_digits.npy", np.array(indices))


    # _, images, _, images_test  = get_mnist_all_classes(10, 1)
    # fig, axes = plt.subplots(10, 10)
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(images[i, ...].reshape(28, 28))
    # plt.show()
    # fig, axes = plt.subplots(10, 1)
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(images_test[i, ...].reshape(28, 28))
    # plt.show()
