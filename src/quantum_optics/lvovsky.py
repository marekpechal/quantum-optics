"""
Implementation of the maximum likelihood density matrix reconstruction
algorithm from Lvovsky 2004
https://arxiv.org/abs/quant-ph/0311097
https://doi.org/10.1088/1464-4266/6/6/014

Marek Pechal, 2021-2024
"""

import numpy as np
import pickle
from quantum_optics.measurements import (
    quadrature_measurement_operator,
    draw_from_pdf,
    )
from quantum_optics.core import (
    expectation,
    fock_wavefunction,
    )

def R_op(rho, theta_arr, x_arr, eta=None):
    """R operator defined by Eq. (2) in Lvovsky 2004.

    Args:
        rho (array): Density matrix.
        theta_arr (array): Array of quadrature angle values.
        x_arr (array): Array of quadrature measurement outcomes.
        eta (float or None, optional): If not None, the projective measurement
            is preceded by a loss channel with transmission eta.
            Defaults to None.

    Returns:
        array: The R operator.
    """
    assert rho.shape[0] == rho.shape[1], "density matrix is not square"
    dim = rho.shape[0]
    res = np.zeros_like(rho)
    for theta, x in zip(theta_arr, x_arr):
        P = quadrature_measurement_operator(dim, theta, x, eta=eta)
        pr = np.trace(P.dot(rho))
        res += P/pr
    return res

def iteration_step(rho, theta_arr, x_arr, eta=None):
    """Iteration step (Eq. (5)) of the algorithm from Lvovsky 2004.

    Args:
        rho (array): Density matrix.
        theta_arr (array): Array of quadrature angle values.
        x_arr (array): Array of quadrature measurement outcomes.
        eta (float or None, optional): If not None, the projective measurement
            is preceded by a loss channel with transmission eta.
            Defaults to None.

    Returns:
        array: Updated density matrix.
    """
    R = R_op(rho, theta_arr, x_arr, eta=eta)
    res = R.dot(rho).dot(R)
    res = res/np.trace(res)
    return res

def generate_data(state, num_quadratures, num_reps,
        x_range, n_subdiv,
        seed=None, eta=None):
    """Make example data for Lvovsky algo.

    Args:
        filename (str): Name of the pickle file to save to.
        state (array): State vector or density matrix of the state.
        num_quadratures (int): Number of quadratures to measure.
        num_reps (int): Number of measurements per quadrature.
        x_range (tuple): Finite range (x1, x2) from which to draw.
        n_subdiv (int): Number of intervals into which to subdivide the range.
        seed (int or None, optional): Seed for the rng. Defaults to None.
        eta (float or None, optional): Quantum efficiency. If None,
            eta = 1. Defaults to None.

    Returns:
        tuple: (theta_arr, x_arr)
            theta_arr is an array of quadrature angles.
            x_arr is an array of quadrature values.
    """

    dim = state.shape[0]
    is_pure = (len(state.shape)==1)
    theta_rng = np.linspace(0, np.pi, num_quadratures+1)[:-1]
    theta_arr = np.array([theta
        for theta in theta_rng
        for _ in range(num_reps)])
    x_arr = []
    for idx, theta in enumerate(theta_rng):
        if is_pure:
            pdf = lambda x: abs(sum([a*fock_wavefunction(n, theta, x)
                for n, a in enumerate(state)]))**2
        else:
            pdf = lambda x: expectation(
                quadrature_measurement_operator(dim, theta, x, eta=eta),
                state).real
        samples = draw_from_pdf(pdf, num_reps, x_range, n_subdiv,
            seed=(seed+idx) if seed is not None else None)
        x_arr = np.concatenate((x_arr, samples))
    return (theta_arr, x_arr)
