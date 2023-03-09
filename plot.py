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

def linear(x, A, b):
    return x @ A + b
            

H1a, H1b, H2a, H2b, H3a, H3b, L1a, L1b, L2a, L2b, E1a, E1b, E2a, E2b, E3a, E3b = read(
)


sigmoid = scipy.special.expit
for R0, path_plot, path_diff in (1, "R1.png", "R1e.png"), (2, "R2.png",
                                                           "R2e.png"):
    n = 1000
    x = np.empty((n, 1))
    x[:, 0] = np.linspace(-10, 10, n)
    y = np.zeros((n, 1))
    z = np.zeros((n, 1))
    R = R0 * np.ones((n, 1))
    r1 = np.sqrt((x - R)**2 + y**2 + z**2)
    r2 = np.sqrt((x + R)**2 + y**2 + z**2)
    f1 = np.exp(-r1)
    f2 = np.exp(-r2)
    h = sigmoid(linear(np.hstack((f1, f2)), H1a, H1b))
    h = sigmoid(linear(h, H2a, H2b))
    h = linear(2 * h, H3a, H3b)
    l = sigmoid(linear(R, L1a, L1b))
    l = linear(l, L2a, L2b)
    e = sigmoid(linear(R, E1a, E1b))
    e = sigmoid(linear(e, E2a, E2b))
    e = linear(e, E3a, E3b)
    psi = f1 + f2 + h * l
    psi = h * l + f1 + f2
    psi_lcao = f1 + f2
    plt.plot(x[:, 0], psi[:, 0], 'b')
    plt.plot(x[:, 0], psi_lcao[:, 0], 'r')
    plt.savefig(path_plot)
    plt.close()
    plt.plot(x[:, 0], psi[:, 0] - psi_lcao[:, 0], 'r')
    plt.savefig(path_diff)
    plt.close()
