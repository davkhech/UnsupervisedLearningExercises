import numpy as np


def random_initialize(data_array, num_clusters):
    return data_array[np.random.choice(len(data_array), num_clusters)]


def plus_plus_initialize(data_array, num_clusters):
    ps = np.ones(len(data_array)) / len(data_array)
    points = np.zeros(shape=(num_clusters, data_array.shape[1]))
    for i in range(num_clusters):
        c = data_array[np.random.choice(len(data_array), 1, p=ps)]
        d = np.zeros(shape=(len(data_array), i + 1))
        for j in range(data_array.shape[1]):
            points[i, j] = c[0, j]
        for j in range(i + 1):
            diff = ((data_array - points[j, :]) / 1000) ** 2
            d[:, j] = np.sum(diff, axis=1)
        mins = np.min(d, axis=1)
        ps = mins / np.sum(mins)

    return points


def loss(data, means, r):
    d = np.zeros_like(r)
    for m_i in range(len(means)):
        diff = (data - means[m_i]) ** 2
        d[:, m_i] = np.sum(diff, axis=1)
    return np.sum(d * r) / len(data)


def expectation(data, means, r):
    d = np.zeros_like(r)
    for m_i in range(len(means)):
        diff = (data - means[m_i]) ** 2
        d[:, m_i] = np.sqrt(np.sum(diff, axis=1))
    r = np.zeros_like(r)
    t = np.argmin(d, axis=1)
    for i in range(len(t)):
        r[i, t[i]] = 1
    return r


def maximization(data, r):
    means = np.zeros(shape=(r.shape[1], data.shape[1]))
    cnts = np.sum(r, axis=0) + 1
    for i in range(len(cnts)):
        for j in range(data.shape[1]):
            mul = data[:, j] * r[:, i]
            means[i, j] = np.sum(mul) / cnts[i]
    return means


class KMeans(object):
    def __init__(self, num_clusters):
        self.K = num_clusters
        self.means = []
        self.r = []

    def initialize(self, data):
        """
        :param data: data, numpy 2-D array
        """
        self.means = plus_plus_initialize(data, self.K)
        self.r = np.zeros(shape=(len(data), self.K))

    def fit(self, data, max_iter=100, precision=1e-6):
        """
        :param data: data to fit, numpy 2-D array
        """
        l = 0
        self.initialize(data)
        for i in range(max_iter):
            self.r = expectation(data, self.means, self.r)
            self.means = maximization(data, self.r)
            cl = loss(data, self.means, self.r)
            if np.abs((cl - l) / cl) < precision:
                break
            else:
                l = cl

    def predict(self, data):
        """
        Return index of the cluster the point is most likely to belong.
        :param data: data, numpy 2-D array
        :return: labels, numpy 1-D array
        """
        cls = expectation(data, self.means, np.zeros(shape=(len(data), self.K)))
        return np.argmax(cls, axis=1)

    def get_centers(self):
        """
        Return list of centers of the clusters, i.e. means
        """
        return self.means
