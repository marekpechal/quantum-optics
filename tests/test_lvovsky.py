from quantum_optics.lvovsky import (
    iteration_step,
    generate_data,
    )
from quantum_optics.core import (
    fock_psi,
    psi_to_rho,
    )
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np

def test_reconstruction_pure():
    num_quadratures = 8
    num_reps = 100
    dim = 5
    seed = 0
    psi = fock_psi(dim, 1)
    theta_arr, x_arr = generate_data(psi, num_quadratures, num_reps,
        (-8.0, 8.0), 2000,
        seed=seed)
    rho = np.eye(dim).astype(complex)
    n_iters = 15
    for _ in range(n_iters):
        rho = iteration_step(rho, theta_arr, x_arr)
    assert (
        abs(np.trace(rho)-1)<1e-6 and
        abs(rho[1, 1]-1)<0.1), \
        "reconstruction check failed"

def test_reconstruction_mixed():
    num_quadratures = 8
    num_reps = 100
    dim = 5
    seed = 0
    psi = fock_psi(dim, 1)
    rho = psi_to_rho(psi)
    theta_arr, x_arr = generate_data(rho, num_quadratures, num_reps,
        (-8.0, 8.0), 2000,
        seed=seed)
    rho = np.eye(dim).astype(complex)
    n_iters = 15
    for _ in range(n_iters):
        rho = iteration_step(rho, theta_arr, x_arr)
    assert (
        abs(np.trace(rho)-1)<1e-6 and
        abs(rho[1, 1]-1)<0.1), \
        "reconstruction check failed"

def test_reconstruction_pure_with_loss():
    num_quadratures = 16
    num_reps = 1000
    dim = 5
    seed = 0
    eta = 0.75
    psi = fock_psi(dim, 1)
    theta_arr, x_arr = generate_data(psi, num_quadratures, num_reps,
        (-8.0, 8.0), 2000,
        seed=seed, eta=eta)
    rho = np.eye(dim).astype(complex)
    n_iters = 25
    for _ in range(n_iters):
        rho = iteration_step(rho, theta_arr, x_arr, eta=eta)
    assert (
        abs(np.trace(rho)-1)<1e-6 and
        abs(rho[1, 1]-1)<0.1), \
        "reconstruction check failed"

if __name__ == "__main__":
    test_reconstruction_pure()
    test_reconstruction_mixed()
    test_reconstruction_pure_with_loss()
