import scipy.special
import numpy as np

def normalize(v):
    """Normalize vector."""
    return np.array(v) / np.linalg.norm(v)

def overlap(u, v):
    """Overlap (scalar product) of two vectors."""
    return np.dot(u.conjugate(), v)

def expectation(operator, state):
    if len(state.shape) == 1:
        return overlap(state, operator @ state)
    else:
        return np.trace(operator @ state)

def destroy(dim):
    """Destruction ladder operator in the Fock basis.

    Args:
        dim (int): Dimension of the Hilbert space.

    Returns:
        array
    """
    return np.diag(np.sqrt(range(1, dim)), 1)

def coherent_psi(dim, alpha):
    """Coherent state as a state vector in the Fock basis.

    Args:
        dim (int): Dimension of the Hilbert space.
        alpha (complex): Displacement of the coherent state.

    Returns:
        array: Pure state vector representing the coherent state.
    """
    return normalize([alpha**j/np.sqrt(scipy.special.factorial(j))
        for j in range(dim)])

def cat_psi(dim, alpha, phi, N_comps=2):
    """N-component cat state.

    Args:
        dim (int): Dimension of the Hilbert space.
        alpha (complex): Displacement of the cat state components.
        phi (float): Phase difference between subsequent component kets.
        N_comps (int, optional): Number of cat components (defaults to 2).

    Returns:
        array: Pure state vector of the cat in the Fock basis. Defined as
            `A sum_{k=0}^{N_comps-1} exp(i k phi)
                |alpha exp(2 pi i k / N_comps)>`,
            where `A` is a normalization coefficient.
    """
    return normalize(
        sum([np.exp(1j*k*phi) *
                coherent_psi(dim, alpha*np.exp(2j*np.pi*k/N_comps))
            for k in range(N_comps)]))

def thermal_rho(dim, n):
    """Thermal state as a density matrix in the Fock basis.

    Args:
        dim (int): Dimension of the Hilbert space.
        n (float): Mean number of photons.

    Returns:
        array: Density matrix of the thermal state in the Fock basis.
    """
    p = (n/(n+1)) ** np.arange(dim)
    return np.diag(p / sum(p))

def psi_to_rho(psi):
    """Convert pure state vector to a density matrix.

    Args:
        psi (array): Pure state vector.

    Returns:
        array: Density matrix.
    """
    return np.outer(psi, psi.conjugate())

def fock_wavefunction(n, theta, x):
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

def apply_loss(state, eta):
    """Apply a loss channel to a state expressed in the Fock basis.

    Args:
        psi (array): State vector or density matrix in the Fock basis.
        eta (float): Transmission of the loss channel.

    Returns:
        array: Density matrix of the resulting state in the Fock basis.
    """
    is_pure = (len(state.shape) == 1)
    dim = state.shape[0]
    a = destroy(dim)
    k = 0
    sk_ak_state = state # this is s^k * a^k @ psi for a pure state or
                        # s^(2k) * a^k @ rho @ ad^k for a mixed state
    s = np.sqrt(1-eta)
    D = np.diag(np.sqrt(eta) ** np.arange(dim))
    rho = np.zeros((dim, dim), dtype=complex)
    while k < dim:
        if is_pure:
            rho += psi_to_rho(D @ sk_ak_state) / scipy.special.factorial(k)
            sk_ak_state = s * a @ sk_ak_state
        else:
            rho += D @ sk_ak_state @ D / scipy.special.factorial(k)
            sk_ak_state = s**2 * a @ sk_ak_state @ a.conj().T
        k += 1
    return rho
