"""
Implementation of the maximum likelihood density matrix reconstruction
algorithm from Lvovsky 2004
https://arxiv.org/abs/quant-ph/0311097
https://doi.org/10.1088/1464-4266/6/6/014

Marek Pechal, 2021-2024
"""

import numpy as np
from quantum_optics.measurements import quadrature_measurement_operator

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
        P = quadrature_measurement_operator(dim, theta, x, eta=eta)
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
