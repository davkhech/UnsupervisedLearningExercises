#!/usr/bin/env python3
"""
Write a Kalman Filter
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import matmul


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='File with the signal to be filtered.',
                        default='noisy_1d.csv')
    args = parser.parse_args(*argument_array)
    return args


class KalmanFilter:

    def __init__(self, measurement_sigma, process_sigma, 
                covariance_prior, location_prior):
        self.measurement_sigma = measurement_sigma
        self.process_sigma = process_sigma
        self.covariance_prior = covariance_prior
        self.location_prior = location_prior

    def step(self, observation, delta_t=1.):
        H = np.array([[1, 0]])
        F = np.array([[1, delta_t], [0, 1]])
        G = np.array([[0.5 * delta_t ** 2], [delta_t]])
        R = np.array([self.measurement_sigma ** 2])
        Q = np.dot(G, G.T) * self.process_sigma ** 2

        next_prediction = np.dot(F, self.location_prior)
        covariance_prior = np.dot(np.dot(F, self.covariance_prior), F.T) + Q

        K = np.dot(np.dot(covariance_prior, H.T), inv(R + np.dot(np.dot(H, covariance_prior), H.T)))
        self.location_prior = self.location_prior + np.dot(K, observation - np.dot(H, self.location_prior))
        self.covariance_prior = np.dot((np.eye(2) - np.dot(K, H)), covariance_prior)
        return self.location_prior[0]


def main(args):
    df = pd.read_csv(args.csv)
    unfiltered = [np.array([row['XX']]) for i, row in df.iterrows()]

    kf = KalmanFilter(100, 0.1, np.array([[1., 0.], [0., 1.]]), np.array([0., 0.]))
    filtered = [kf.step(x) for x in unfiltered]
    
    plt.plot(unfiltered)
    plt.plot(filtered[1:])
    plt.show()


if __name__ == '__main__':
  args = parse_args()
  main(args)
