import numpy as np
from quantum_optics.measurements import (
    quadrature_measurement_operator,
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


if __name__ == "__main__":
    test_quadrature_measurement_operator()
