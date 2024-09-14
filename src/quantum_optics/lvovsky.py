"""
Implementation of the maximum likelihood density matrix reconstruction
algorithm from Lvovsky 2004
https://arxiv.org/abs/quant-ph/0311097
https://doi.org/10.1088/1464-4266/6/6/014

Marek Pechal, 2021-2024
"""

import functools
import numpy as np
import scipy.special
import qutip as qt

def fock_overlap(n, theta, x):
    """Overlap <x_theta|n> between a Fock state and eigenstate of a
    quadrature defined by angle theta.

    Args:
        n (int): Fock state number.
        theta (float): Angle of quadrature.
        x (float): Eigenvalue describing quadrature eigenstate.

    Returns:
        float: Overlap <x_theta|n>.
    """
    Hn = scipy.special.hermite(n)
    return (np.exp(1j*n*theta) * (2/np.pi)**0.25 * Hn(x*np.sqrt(2)) *
        np.exp(-x**2) / np.sqrt(2**n*scipy.special.factorial(n)))

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

def measurement_operator(dim, theta, x, eta=None):
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
    v = np.array([fock_overlap(n, theta, x) for n in range(dim)])
    res = np.outer(v, v.conjugate()) # lossless measurement operator

    if eta is None:        # omitted eta means eta=1.0
        L = np.eye(dim**2) # the map from lossless to lossy with eta=1.0 is just the identity
    else:
        L = loss_map(dim, eta)

    res = L.dot(res.flatten()).reshape((dim, dim)) # apply the lossless->lossy map
    return res

def R_op(rho, theta_arr, x_arr, eta=None):
    """R operator defined by Eq. (2) in Lvovsky 2004.

    Args:
        rho (array): Density matrix.
        theta_arr (array): Array of quadrature angle values.
        x_arr (array): Array of quadrature measurement outcomes.
        eta (float or None, optional): If not None, the projective measurement
            is preceded by a loss channel with transmission eta.
            Defaults to None.
    """
    assert rho.shape[0] == rho.shape[1], "density matrix is not square"
    dim = rho.shape[0]
    res = np.zeros_like(rho)
    for theta, x in zip(theta_arr, x_arr):
        P = measurement_operator(dim, theta, x, eta=eta)
        pr = np.trace(P.dot(rho))
        res += P/pr
    return res

def iteration_step(rho, theta_arr, x_arr, eta=None):
    """Iteration step (Eq. (5)) of the algorithm from Lvovsky 2004.

    rho (array): Density matrix.
    theta_arr (array): Array of quadrature angle values.
    x_arr (array): Array of quadrature measurement outcomes.
    eta (float or None, optional): If not None, the projective measurement
        is preceded by a loss channel with transmission eta.
        Defaults to None.
    """
    R = R_op(rho, theta_arr, x_arr, eta=eta)
    res = R.dot(rho).dot(R)
    res = res/np.trace(res)
    return res
