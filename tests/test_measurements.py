import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from quantum_optics.core import (
    cat_psi,
    fock_psi,
    thermal_rho,
    )
from quantum_optics.measurements import (
    quadrature_measurement_operator,
    husimi_Q,
    wigner_W,
    )

def test_quadrature_measurement_operator():
    dim = 5
    theta = 0.0
    eta = 0.5

    xrng = np.linspace(-3, 3, 101)
    prng1 = [quadrature_measurement_operator(dim, theta, x)[0, 0]
        for x in xrng]
    prng2 = [quadrature_measurement_operator(dim, theta+np.pi/2, x)[0, 0]
        for x in xrng]
    prng_th = [np.exp(-2*x**2)/np.sqrt(np.pi/2) for x in xrng]
    np.testing.assert_array_almost_equal(prng1, prng_th,
        err_msg="measurement pdf check failed")
    np.testing.assert_array_almost_equal(prng2, prng_th,
        err_msg="measurement pdf check failed")

    prng = [quadrature_measurement_operator(dim, theta, x)[1, 1]
        for x in xrng]
    prng_th = [4*x**2*np.exp(-2*x**2)/np.sqrt(np.pi/2) for x in xrng]
    np.testing.assert_array_almost_equal(prng, prng_th,
        err_msg="measurement pdf check failed")

    prng = [quadrature_measurement_operator(dim, theta, x, 0.5)[1, 1]
        for x in xrng]
    prng_th = [(4*x**2+1)*np.exp(-2*x**2)/(np.sqrt(2*np.pi)) for x in xrng]
    np.testing.assert_array_almost_equal(prng, prng_th,
        err_msg="measurement pdf check failed")

def test_husimi_Q():
    dim = 40
    n = 1
    psi = fock_psi(dim, n)
    xrng = np.linspace(-5., 5., 101)

    # test with Fock state against analytical calculation
    qvals = np.array([[husimi_Q(psi, x+1j*y) for x in xrng] for y in xrng])
    qvals_th = np.array([[
        np.exp(-(x**2+y**2))*(x**2+y**2)**n
        for x in xrng] for y in xrng])/scipy.special.factorial(n)
    np.testing.assert_array_almost_equal(qvals, qvals_th,
        decimal=5,
        err_msg="Husimi Q test failed")

    dx = xrng[1]-xrng[0]
    assert abs(sum(qvals.flatten())*dx**2-np.pi)<1e-4, \
        "Husimi Q normalization test failed"

    # test with thermal state against analytical calculation
    n = 2.0
    rho = thermal_rho(dim, n)
    qvals = np.array([[husimi_Q(rho, x+1j*y) for x in xrng] for y in xrng])
    qvals_th = np.array([[
        np.exp(-(x**2+y**2)/(n+1))
        for x in xrng] for y in xrng])/(n+1)
    np.testing.assert_array_almost_equal(qvals, qvals_th,
        decimal=5,
        err_msg="Husimi Q test failed")

def test_wigner_W():
    dim = 30
    alpha = 0.7+0.7j
    psi = cat_psi(dim, alpha, 0.0)
    xrng = np.linspace(0.0, 1.5, 6)

    # test with cat state against analytical calculation
    wvals = np.array([[wigner_W(psi, x+1j*y) for x in xrng] for y in xrng])
    wvals_th = np.array([[
        np.exp(-2*((x-alpha.real)**2+(y-alpha.imag)**2)) +
        np.exp(-2*((x+alpha.real)**2+(y+alpha.imag)**2)) +
        2*np.exp(-2*(x**2+y**2))*np.cos(4*(alpha*(x-1j*y)).imag)
        for x in xrng] for y in xrng])/(2*(1+np.exp(-2*abs(alpha)**2)))
    np.testing.assert_array_almost_equal(wvals, wvals_th,
        err_msg="Wigner W test failed")

if __name__ == "__main__":
    test_quadrature_measurement_operator()
    test_husimi_Q()
    test_wigner_W()
