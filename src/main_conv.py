import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits

from conv_1d_kernels import CConvKernelMovingAverage, CConvKernelTriangle

x, y = load_mnist_data()

z = x[0, :]

plt.imshow(z.reshape(28,28))
plt.show()

# conv = CConvKernelMovingAverage()
conv = CConvKernelTriangle(kernel_size=15)
conv.kernel_size = 5
zp = conv.kernel(z)


plt.imshow(zp.reshape(28,28))
plt.show()

