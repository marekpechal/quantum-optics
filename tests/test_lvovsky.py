from quantum_optics.lvovsky import (
    fock_overlap,
    measurement_operator,
    iteration_step
    )
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np

def test_fock_overlap():
    assert (
        abs(fock_overlap(1, 0.0, 0.0))<1e-6 and
        abs(fock_overlap(3, 1.234, 0.0))<1e-6 and
        abs(fock_overlap(3, 1.234, 3.57)+fock_overlap(3, 1.234, -3.57))<1e-6 and
        abs(fock_overlap(4, 1.234, 3.57)-fock_overlap(4, 1.234, -3.57))<1e-6), \
        "parity check failed"

    assert (
        abs(scipy.integrate.quad(
            lambda x: abs(fock_overlap(1, 0.0, x))**2,
            -10.0, 10.0)[0] - 1)<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: abs(fock_overlap(2, 2.781, x))**2,
            -10.0, 10.0)[0] - 1)<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: abs(fock_overlap(3, -1.57, x))**2,
            -10.0, 10.0)[0] - 1)<1e-6), \
        "normalization check failed"

    assert (
        abs(scipy.integrate.quad(
            lambda x: (fock_overlap(0, 0.0, x).conjugate()
                *fock_overlap(2, 0.0, x)).real,
            -10.0, 10.0)[0])<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: (fock_overlap(0, 0.0, x).conjugate()
                *fock_overlap(2, 0.0, x)).imag,
            -10.0, 10.0)[0])<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: (fock_overlap(1, 3.456, x).conjugate()
                *fock_overlap(3, 2.456, x)).real,
            -10.0, 10.0)[0])<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: (fock_overlap(1, 2.456, x).conjugate()
                *fock_overlap(3, 2.456, x)).imag,
            -10.0, 10.0)[0])<1e-6), \
        "orthogonality check failed"

def test_measurement_operator():
    dim = 5
    theta = 0.0
    eta = 0.5

    xrng = np.linspace(-3, 3, 101)
    prng1 = [measurement_operator(dim, theta, x)[0, 0] for x in xrng]
    prng2 = [measurement_operator(dim, theta+np.pi/2, x)[0, 0] for x in xrng]
    prng_th = [np.exp(-2*x**2)/np.sqrt(np.pi/2) for x in xrng]
    np.testing.assert_array_almost_equal(prng1, prng_th,
        err_msg="measurement pdf check failed")
    np.testing.assert_array_almost_equal(prng2, prng_th,
        err_msg="measurement pdf check failed")

    prng = [measurement_operator(dim, theta, x)[1, 1] for x in xrng]
    prng_th = [4*x**2*np.exp(-2*x**2)/np.sqrt(np.pi/2) for x in xrng]
    np.testing.assert_array_almost_equal(prng, prng_th,
        err_msg="measurement pdf check failed")

    prng = [measurement_operator(dim, theta, x, 0.5)[1, 1] for x in xrng]
    prng_th = [(4*x**2+1)*np.exp(-2*x**2)/(np.sqrt(2*np.pi)) for x in xrng]
    np.testing.assert_array_almost_equal(prng, prng_th,
        err_msg="measurement pdf check failed")

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
    test_fock_overlap()
    test_measurement_operator()
    test_reconstruction()
