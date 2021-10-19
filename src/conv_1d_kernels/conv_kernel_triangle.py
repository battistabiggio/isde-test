from conv_1d_kernels import CConvKernel
import numpy as np


class CConvKernelTriangle(CConvKernel):

    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)

    def kernel_mask(self):
        self._mask = np.ones(self.kernel_size)
        self._mask = self._mask.cumsum()
        k = (self.kernel_size - 1) // 2
        self._mask[-k:] = np.flip(self._mask[:k])

