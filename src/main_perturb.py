import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from data_perturb import CDataPerturbRandom

x, y = load_mnist_data()

# implementing perturb_dataset(x) --> xp (perturbed dataset)
# initialize Xp
# loop over the rows of X, then at each iteration:
#    extract the given row,
#    apply the data_perturbation function
#    copy the result (perturbed image) in xp

data_pert = CDataPerturbRandom()
xp = data_pert.perturb_dataset(x)
# plot_ten_digits(x, y)
# plot_ten_digits(xp, y)

# split MNITS data into 60% training and 40% test sets
n_tr = int(0.6 * x.shape[0])
print("Number of total samples: ",x.shape[0],
      "\nNumber of training samples: ", n_tr)

x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

clf = SVC(kernel='linear')
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_ts)
clf_acc = np.mean(y_ts == y_pred)
print("Test accuracy: ", int(clf_acc*10000)/100, "%")

k_values = np.array([0, 10, 20, 50, 100, 200, 300, 400, 500])
test_accuracies = np.zeros(shape=k_values.shape)
for i, k in enumerate(k_values):
    # perturb ts
    data_pert.K = k
    xp = data_pert.perturb_dataset(x_ts)
    # plot_ten_digits(xp, y)
    # compute predicted labels on the perturbed ts
    y_pred = clf.predict(xp)
    # compute classification accuracy using y_pred
    clf_acc = np.mean(y_ts == y_pred)
    print("Test accuracy(K=", k, "): ", int(clf_acc * 10000) / 100, "%")
    test_accuracies[i] = clf_acc

plt.plot(k_values, test_accuracies)
plt.xlabel('K')
plt.ylabel('Test accuracy(K)')
plt.show()
