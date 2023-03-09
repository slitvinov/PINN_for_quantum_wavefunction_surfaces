import numpy as np
import scipy.special
import matplotlib.pylab as plt


def read():
    dtype = np.dtype("float64")
    with open("model.bin", "rb") as file:
        while True:
            ndim = int.from_bytes(file.read(4), "little")
            if ndim == 0:
                break
            shape = [
                int.from_bytes(file.read(4), "little") for d in range(ndim)
            ]
            size = np.product(shape) * dtype.itemsize
            yield np.ndarray(shape, dtype, file.read(size))


H1a, H1b, H2a, H2b, H3a, H3b, L1a, L1b, L2a, L2b, E1a, E1b, E2a, E2b, E3a, E3b = read(
)

Rlo = 0.2
Rhi = 4.0
sigmoid = scipy.special.expit
n = 1000
R = np.empty((n, 1))
R[:, 0] = np.linspace(Rlo, Rhi, n)
e = sigmoid(R @ E1a + E1b)
e = sigmoid(e @ E2a + E2b)
e = e @ E3a + E3b
E = e + 2 / R
plt.plot(R[:, 0], E[:, 0], 'b')
plt.savefig("energy.png")
