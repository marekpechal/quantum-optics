import numpy as np
import functools
import scipy.special
from quantum_optics.core import (
    overlap,
    expectation,
    fock_wavefunction,
    displaced_fock_psi,
    coherent_psi,
    )

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

def husimi_Q(state, alpha, dim=None):
    """Husimi Q probability function

    Args:
        state (array): Pure state vector or density matrix.
        alpha (complex): Displacement at which to evaluate Q.
        dim (int or None, optional): Dimension of the Hilbert space.
            If None, it is determined from the shape of `state`. Otherwise,
            `state` is extended to the given dimension.

    Returns:
        complex: Value of Q(alpha).
    """
    state = state.copy()
    if dim is None:
        dim = state.shape[0]
    if len(state.shape) == 1:
        state.resize(dim)
        return abs(overlap(coherent_psi(dim, alpha), state))**2
    else:
        state.resize((dim, dim))
        return expectation(state, coherent_psi(dim, alpha)).real

def wigner_W(state, alpha, dim=None):
    """Wigner W quasi-probability function

    Args:
        state (array): Pure state vector or density matrix.
        alpha (complex): Displacement at which to evaluate W.
        dim (int or None, optional): Dimension of the Hilbert space.
            If None, it is determined from the shape of `state`. Otherwise,
            `state` is extended to the given dimension.

    Returns:
        complex: Value of W(alpha).
    """
    state = state.copy()
    if dim is None:
        dim = state.shape[0]
    if len(state.shape) == 1:
        state.resize(dim)
        return sum([
            (-1)**n * abs(overlap(displaced_fock_psi(dim, n, alpha), state))**2
            for n in range(dim)])
    else:
        state.resize((dim, dim))
        return sum([
            (-1)**n * expectation(state, displaced_fock_psi(dim, n, alpha))
            for n in range(dim)])

def draw_from_pdf(pdf, N, x_range, n_subdiv, seed=None):
    """Draw random numbers from a given probability distribution.

    The pdf is restricted to a finite range which is the subdivided into
    equally sized subintervals. The pdf is replaced by a piecewise constant
    function on the subintervals (with value taken from the mid-point)
    and rescaled to properly normalize it. The returned values are drawn
    from this modified distribution.

    Args:
        pdf (callable): Probability density function as a callable taking a
            single float argument and returning a single float. Does not need
            to be normalized.
        N (int): Number of i.i.d. values to draw.
        x_range (tuple): Finite range (x1, x2) from which to draw.
        n_subdiv (int): Number of intervals into which to subdivide the range.
        seed (int or None, optional): Seed for the rng. Defaults to None.
    """
    rng = np.random.default_rng(seed=seed)
    dx = (x_range[1]-x_range[0])/n_subdiv
    x_ctr_arr = dx*(np.arange(n_subdiv)+0.5)+x_range[0]
    x_arr = dx*np.arange(n_subdiv+1)+x_range[0]
    p_ctr_arr = np.array([pdf(x) for x in x_ctr_arr])
    p_ctr_arr = p_ctr_arr / sum(p_ctr_arr)
    F = scipy.interpolate.interp1d(
        np.concatenate(([0], np.cumsum(p_ctr_arr))), x_arr)
    return np.array([F(rng.uniform()) for _ in range(N)])
