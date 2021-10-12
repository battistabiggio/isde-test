import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from data_perturb import CDataPerturbUniform, CDataPerturbGaussian


def robustness_test(clf, data_pert, param_name, param_values):
    test_accuracies = np.zeros(shape=param_values.shape)
    for i, k in enumerate(param_values):
        setattr(data_pert, param_name, k)  # data_pert.sigma = k
        xp = data_pert.perturb_dataset(x_ts)
        # plot_ten_digits(xp, y)
        # compute predicted labels on the perturbed ts
        y_pred = clf.predict(xp)
        # compute classification accuracy using y_pred
        clf_acc = np.mean(y_ts == y_pred)
        print("Test accuracy(K=", k, "): ", int(clf_acc * 10000) / 100, "%")
        test_accuracies[i] = clf_acc
    return test_accuracies


x, y = load_mnist_data()

# implementing perturb_dataset(x) --> xp (perturbed dataset)
# initialize Xp
# loop over the rows of X, then at each iteration:
#    extract the given row,
#    apply the data_perturbation function
#    copy the result (perturbed image) in xp

# split MNITS data into 60% training and 40% test sets
n_tr = int(0.6 * x.shape[0])
print("Number of total samples: ", x.shape[0],
      "\nNumber of training samples: ", n_tr)

x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

clf = NMC()
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_ts)
clf_acc = np.mean(y_ts == y_pred)
print("Test accuracy: ", int(clf_acc * 10000) / 100, "%")

param_values = np.array([0, 10, 20, 50, 100, 200, 300, 400, 500])


plt.figure(figsize=(10,5))
test_accuracies = robustness_test(
    clf,  CDataPerturbUniform(), param_name='K', param_values=param_values)

plt.subplot(1, 2, 1)
plt.plot(param_values, test_accuracies)
plt.xlabel('K')
plt.ylabel('Test accuracy(K)')

test_accuracies = robustness_test(
    clf,  CDataPerturbGaussian(), param_name='sigma', param_values=param_values)

plt.subplot(1, 2, 2)
plt.plot(param_values, test_accuracies)
plt.xlabel('sigma')
plt.ylabel('Test accuracy(sigma)')

plt.show()
