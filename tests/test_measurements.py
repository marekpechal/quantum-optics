import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from quantum_optics.core import (
    normalize,
    cat_psi,
    fock_psi,
    thermal_rho,
    fock_wavefunction,
    expectation,
    destroy,
    )
from quantum_optics.measurements import (
    quadrature_measurement_operator,
    husimi_Q,
    wigner_W_naive,
    wigner_W,
    draw_from_pdf,
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

def test_wigner_W_naive():
    dim = 30
    alpha = 0.7+0.7j
    psi = cat_psi(dim, alpha, 0.0)
    xrng = np.linspace(0.0, 1.5, 6)

    # test with cat state against analytical calculation
    wvals = np.array([[wigner_W_naive(psi, x+1j*y)
        for x in xrng] for y in xrng])
    wvals_th = np.array([[
        np.exp(-2*((x-alpha.real)**2+(y-alpha.imag)**2)) +
        np.exp(-2*((x+alpha.real)**2+(y+alpha.imag)**2)) +
        2*np.exp(-2*(x**2+y**2))*np.cos(4*(alpha*(x-1j*y)).imag)
        for x in xrng] for y in xrng])/(2*(1+np.exp(-2*abs(alpha)**2)))
    np.testing.assert_array_almost_equal(wvals, wvals_th,
        err_msg="Wigner W test failed")

def test_wigner_W():
    dim = 30
    alpha = 0.7+0.7j
    psi = cat_psi(dim, alpha, 0.0)
    xrng = np.linspace(0.0, 1.5, 6)

    # test with cat state against analytical calculation
    wvals = np.array([[wigner_W(psi, x+1j*y)
        for x in xrng] for y in xrng])
    wvals_th = np.array([[
        np.exp(-2*((x-alpha.real)**2+(y-alpha.imag)**2)) +
        np.exp(-2*((x+alpha.real)**2+(y+alpha.imag)**2)) +
        2*np.exp(-2*(x**2+y**2))*np.cos(4*(alpha*(x-1j*y)).imag)
        for x in xrng] for y in xrng])/(2*(1+np.exp(-2*abs(alpha)**2)))
    np.testing.assert_array_almost_equal(wvals, wvals_th,
        err_msg="Wigner W test failed")

def test_draw_from_pdf(plot=False):
    seed = 0
    N = 10000
    xmax = 8.0
    n_subdiv = 1000
    pdf = lambda x: np.exp(-x**2/2)
    samples = draw_from_pdf(pdf, N, (-xmax, xmax), n_subdiv, seed=seed)
    assert abs(np.mean(samples))<1e-2, "mean check failed"
    assert abs(np.std(samples)-1)<1e-2, "standard deviation check failed"

    reps = 5
    dim = 5
    N = 100000
    for idx in range(reps):
        rng = np.random.default_rng(seed=2*idx)
        state = rng.normal(size=dim) + 1j*rng.normal(size=dim)
        state = normalize(state)
        pdf = lambda x: abs(sum([a*fock_wavefunction(n, 0, x)
            for n, a in enumerate(state)]))**2
        samples = draw_from_pdf(pdf, N, (-xmax, xmax), n_subdiv, seed=2*idx+1)

        dim_calc = dim+1
        state = state.copy()
        state.resize(dim_calc)
        a = destroy(dim_calc)
        x_op = 0.5*(a+a.conj().T)
        mean_x2 = np.mean(samples**2)
        mean_x2_th = expectation(x_op @ x_op, state).real

        if plot:
            print(f"trial {idx+1} out of {reps}:")
            print(f"mean x^2 from sampled points = {mean_x2}")
            print(f"<psi|X^2|psi> = {mean_x2_th}")
            print(f"deviation = {mean_x2-mean_x2_th}")
            n_bins = 100
            xrng = np.linspace(-xmax, xmax, 401)
            plt.plot(xrng, [(2*xmax*N/n_bins)*pdf(x) for x in xrng])
            plt.hist(samples, n_bins, range=(-xmax, xmax))
            plt.show()
            plt.close()

        assert abs(mean_x2-mean_x2_th)<1e-2, \
            "x^2 expectation value check failed"

if __name__ == "__main__":
    test_quadrature_measurement_operator()
    test_husimi_Q()
    test_wigner_W_naive()
    test_wigner_W()
    test_draw_from_pdf(plot=True)
