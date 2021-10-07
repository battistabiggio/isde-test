import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from utils import load_mnist_data, split_data, plot_ten_digits


class NMC:
    """Class implementing the Nearest Mean Centroid (NMC)
    classification algorithm."""
    def __init__(self):
        self._centroids = None  # init centroids

    @property
    def centroids(self):
        return self._centroids

    def fit(self, x_tr, y_tr):
        """Fit the model to the data (estimating centroids)"""
        n_classes = np.unique(y_tr).size
        n_features = x_tr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k in range(n_classes):
            # extract only images of 0 from x_tr
            xk = x_tr[y_tr == k, :]
            self._centroids[k, :] = np.mean(xk, axis=0)
        return self

    def predict(self, x_ts):
        dist = pairwise_distances(x_ts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred

x, y = load_mnist_data()  #plot_ten_digits(x, y)
x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=1000)
clf = NMC()
clf.fit(x_tr, y_tr)
plot_ten_digits(clf.centroids)
ypred = clf.predict(x_ts)



