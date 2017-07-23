import numpy as np
from scipy.stats import multivariate_normal
from kmeans import KMeans


def multi_normal_pdf(x, mean, covariance):
    """
    Evaluates Multivariate Gaussian Distribution density function
    :param x: location where to evaluate the density function
    :param mean: Center of the Gaussian Distribution
    :param covariance: Covariance of the Gaussian Distribution
    :return: density function evaluated at point x
    """
    var = multivariate_normal(mean=mean, cov=covariance)
    return var.pdf(x)


def expectation(data, weights, means, covariances):
    r = np.zeros(shape=(len(data), len(means)))
    for j in range(len(means)):
        r[:, j] = weights[j] * multi_normal_pdf(data, means[j], covariances[j])
    divisor = np.sum(r, axis=1)
    for i in range(r.shape[1]):
        r[:, i] = r[:, i] / divisor
    return r


def maximization(data, r, means):
    rs = np.sum(r, axis=0)
    weights = rs / len(data)
    centers = np.zeros_like(means)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            centers[i, j] = np.sum(r[:, i] * data[:, j], axis=0) / rs[i]
    covariances = np.zeros(shape=(len(means), data.shape[1], data.shape[1]))
    for j in range(len(means)):
        for i in range(len(data)):
            diff = np.reshape(data[i] - means[j], newshape=(len(means[j]), 1))
            mul = r[i, j] * diff.dot(diff.T)
            covariances[j] += mul
        covariances[j] = covariances[j] / rs[j]
    return weights, centers, covariances


def loss(data, means, r):
    d = np.zeros_like(r)
    for m_i in range(len(means)):
        diff = (data - means[m_i]) ** 2
        d[:, m_i] = np.sum(diff, axis=1)
    return np.sum(d * r) / len(data)


class GaussianMixtureModel(object):
    def __init__(self, num_mixtures):
        self.K = num_mixtures
        self.centers = []  # List of centers
        self.weights = []  # List of weights
        self.covariances = []  # List of covariances
        self.r = None  # Matrix of responsibilities, i.e. gamma

    def initialize(self, data):
        """
        :param data: data, numpy 2-D array
        """
        clf = KMeans(self.K)
        clf.fit(data, 10)
        self.centers = clf.get_centers()
        self.weights = np.ones(self.K) / self.K
        self.covariances = np.array([1e10 * np.eye(data.shape[1]) for _ in range(self.K)])

    def fit(self, data, max_iter=100, precision=1e-6):
        """
        :param data: data to fit, numpy 2-D array
        """
        print 'Gmm: fit'
        self.initialize(data)

        l = 1e12
        for iteration in range(1, max_iter + 1):
            print 'Iteration: %d' % iteration
            self.r = expectation(data, self.weights, self.centers, self.covariances)
            self.weights, self.centers, self.covariances = maximization(data, self.r, self.centers)
            cl = loss(data, self.centers, self.r)
            if np.abs((cl - l) / cl) < precision:
                break
            else:
                print cl, (cl - l) / cl
                l = cl

    def get_centers(self):
        return self.centers

    def get_covariances(self):
        return self.covariances

    def get_weights(self):
        return self.weights

    def predict_cluster(self, data):
        """
        Return index of the clusters that each point is most likely to belong.
        :param data: data, numpy 2-D array
        :return: labels, numpy 1-D array
        """
        print 'Gmm: predict'
        exps = expectation(data, self.weights, self.centers, self.covariances)
        return np.argmax(exps, axis=1)
