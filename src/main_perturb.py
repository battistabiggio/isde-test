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

plt.imshow(x[0,:].reshape(28,28))
plt.show()
plt.imshow(x[1,:].reshape(28,28))
plt.show()

plt.imshow(xp[0,:].reshape(28,28))
plt.show()
plt.imshow(xp[1,:].reshape(28,28))
plt.show()

plt.imshow(x[0,:].reshape(28,28))
plt.show()
plt.imshow(x[1,:].reshape(28,28))
plt.show()

