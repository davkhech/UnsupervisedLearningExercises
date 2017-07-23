#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from kmeans import KMeans
from mixture import GaussianMixtureModel


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_csv', default='../data/data/csv')
    parser.add_argument('--num-clusters', type=int, default=15)
    parser.add_argument('--algorithm', choices=['k-means', 'gmm'],
                        default='k-means')
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    df = pd.read_csv(args.data_csv)
    data = np.array(df[['X', 'Y']])
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], s=3, color='blue')

    if args.algorithm == 'gmm':
        gmm = GaussianMixtureModel(args.num_clusters)
        gmm.fit(data)
        y = gmm.predict_cluster(data)
    else:
        km = KMeans(args.num_clusters)
        km.fit(data)
        y = km.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
