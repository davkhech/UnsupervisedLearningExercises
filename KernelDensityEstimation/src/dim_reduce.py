#!/usr/bin/env python3
"""
This is a boilerplate file for you to get started on MNIST dataset and run SVD.

This file has code to read labels and data from .gz files you can download from
http://yann.lecun.com/exdb/mnist/

Files will work if train-images-idx3-ubyte.gz file and
train-labels-idx1-ubyte.gz files are in the same directory as this
python file.
"""
from __future__ import print_function
import argparse
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist-train-data',
                        default='train-images-idx3-ubyte.gz',
                        help='Path to train-images-idx3-ubyte.gz file '
                        'downloaded from http://yann.lecun.com/exdb/mnist/')
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    # Read data file into numpy matrices
    with gzip.open(args.mnist_train_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        all_data = np.array([np.array(struct.unpack('>{}B'.format(rows * columns),
                                           in_gzip.read(rows * columns)))
                    for _ in range(60000)])

    svd = TruncatedSVD(n_components=5, n_iter=100)
    svd.fit(all_data)

    f, axarr = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for i in range(5):
        axarr[i].set_aspect('equal')
        axarr[i].set_title('Component {}'.format(i + 1))
        axarr[i].imshow(svd.components_[i].reshape(28, 28), cmap='gray')

    plt.tight_layout()
    name = 'Davit Khechoyan'
    plt.savefig('comps-{}.png'.format(name), dpi=320)


if __name__ == '__main__':
    args = parse_args()
    main(args)
