from quantum_optics.lvovsky import (
    iteration_step,
    )
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np

def _generate_random_numbers(pdf, N, xMax=8.0, seed=None):
    rng = np.random.default_rng(seed=seed)
    xRng = np.linspace(-xMax, xMax, 2001)
    pRng = np.array([pdf(x) for x in xRng])
    pRng = pRng/sum(pRng)
    F = scipy.interpolate.interp1d(np.cumsum(pRng), xRng)
    return np.array([F(rng.uniform()) for _ in range(N)])

def test_reconstruction():
    theta_rng = np.linspace(0, np.pi/2, 9)
    reps = 50
    dim = 5
    seed = 0
    theta_arr = np.array([theta for theta in theta_rng for _ in range(reps)])
    x_arr = _generate_random_numbers(
        lambda x: 4*x**2*np.exp(-2*x**2)/np.sqrt(np.pi/2),
        len(theta_arr)*reps,
        seed=seed
        )
    rho = np.eye(dim).astype(complex)
    n_iters = 50
    for _ in range(n_iters):
        rho = iteration_step(rho, theta_arr, x_arr)
    assert (
        abs(np.trace(rho)-1)<1e-6 and
        abs(rho[1, 1]-1)<0.05), \
        "reconstruction check failed"

if __name__ == "__main__":
    test_reconstruction()
