"""
Kernel Density Estimator
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KDE(object):
    """
    use Epanechnikov kernel
    """

    def __init__(self):
        self.h = 1
        self.N = 1
        self.points = []

    def fit(self, data):
        """
        :param data: data to fit, numpy 2-D array
        """
        std = np.std(data, axis=0)
        self.N = len(data)
        self.h = std * np.power((8. * np.sqrt(np.pi) / (3. * self.N)), .2)
        self.points = data

    def log_density(self, data):
        """
        :param data: data to predict density for, numpy 2-D array
        :return: numpy 1-D array, with natural log density of each point
        provided.
        """
        f = lambda x: 3. / (4. * self.N * self.h) * np.sum(1 - np.square(np.minimum(np.abs(x - self.points[:]), 1) / self.h))

        ans = np.zeros_like(data, dtype=np.float)
        for i in range(len(data)):
            ans[i] = f(data[i])
        return ans


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='readings.csv')
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    df = pd.read_csv(args.data)
    X = np.array(df[['reading']])
    plt.hist(X, bins=20)

    kde = KDE()
    kde.fit(X)

    data = np.arange(0, 30, 0.1)
    density = kde.log_density(data)
    new_density = (density - min(density)) * 100000
    print(np.sum(density))
    plt.plot(data, new_density)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
