from conv_1d_kernels import CConvKernel
import numpy as np


class CConvKernelMovingAverage(CConvKernel):

    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)

    def kernel_mask(self):
        self._mask = 1 / self.kernel_size * np.ones(shape=(self.kernel_size,))
