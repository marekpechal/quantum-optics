import numpy as np
import functools
import scipy.special
from quantum_optics.core import fock_wavefunction

@functools.lru_cache(maxsize=None)
def loss_map(dim, eta):
    """Get the dim^2 x dim^2 matrix mapping a lossless measurement operator
    to a lossy one. This map only depends on eta and so can be precalculated.
    Here it's done by caching the function.

    Args:
        dim (int): Dimension of the Hilbert space.
        eta (float): Transmission of the loss channel.

    Returns:
        array: Superoperator mapping a measurement operator onto one where
            the measurement is preceded by application of a loss channel.
            The superoperator is a dim^2 x dim^2 matrix which acts on the
            vector obtained by flattening the measurement operator.
    """
    def B(r, s, eta):
        return np.sqrt(scipy.special.binom(r, s)*eta**s*(1-eta)**(r-s))

    res = np.zeros((dim**2, dim**2), dtype=complex)
    for m in range(dim):
        for n in range(dim):
            idx1 = n*dim + m
            for k in range(min(dim-m, dim-n)):
                idx2 = (n+k)*dim + (m+k)
                res[idx2, idx1] += B(m+k, m, eta)*B(n+k, n, eta)
    return res

def quadrature_measurement_operator(dim, theta, x, eta=None):
    """Measurement operator for a projective quadrature measurement.

    Args:
        dim (int): Dimension of the Hilbert space.
        theta (float): Angle of quadrature.
        x (float): Quadrature value (measurement outcome).
        eta (float or None, optional): If not None, the projective measurement
            is preceded by a loss channel with transmission eta.
            Defaults to None.

    Returns:
        array: The measurement operator in Fock basis.
    """
    v = np.array([fock_wavefunction(n, theta, x) for n in range(dim)])
    res = np.outer(v, v.conjugate()) # lossless measurement operator

    if eta is None:        # omitted eta means eta=1.0
        L = np.eye(dim**2) # the map from lossless to lossy with eta=1.0
    else:                  # is just the identity
        L = loss_map(dim, eta)

    res = L.dot(res.flatten()).reshape((dim, dim)) # apply the
    return res                                     # lossless -> lossy map
