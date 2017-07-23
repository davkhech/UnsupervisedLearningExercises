#!/usr/bin/env python3
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from mixture import GaussianMixtureModel


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_csv')
    parser.add_argument('--num-clusters', type=int, default=15)
    args = parser.parse_args(*argument_array)
    return args


def get_ellipse_from_covariance(matrix, std_multiplier=2):
    values, vectors = np.linalg.eig(matrix)
    maxI = np.argmax(values)
    large, small = values[maxI], values[1 - maxI]
    return (std_multiplier * np.sqrt(large),
            std_multiplier * np.sqrt(small),
            np.rad2deg(np.arccos(vectors[0, 0])))


def main(args):
    df = pd.read_csv(args.data_csv)
    data = np.array(df[['X', 'Y']])
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], s=3, color='blue')

    gmm = GaussianMixtureModel(args.num_clusters)
    gmm.fit(data)
    centers = gmm.get_centers()
    sigmas = gmm.get_covariances()
    weights = gmm.get_weights()

    # Plot ellipses for each of covariance matrices.
    for k in range(len(sigmas)):
        h, w, angle = get_ellipse_from_covariance(sigmas[k])
        e = patches.Ellipse(centers[k], w, h, angle=angle)
        e.set_alpha(np.power(weights[k], .3))
        e.set_facecolor('red')
        plt.axes().add_artist(e)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
