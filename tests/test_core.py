import scipy.integrate
import numpy as np
import qutip as qt
from quantum_optics.core import (
    fock_wavefunction,
    normalize,
    overlap,
    destroy,
    expectation,
    psi_to_rho,
    coherent_psi,
    displaced_fock_psi,
    cat_psi,
    thermal_rho,
    qt_correlated_thermal_rho,
    apply_loss,
    )

def test_fock_wavefunction():
    assert (
        abs(fock_wavefunction(1, 0.0, 0.0))<1e-6 and
        abs(fock_wavefunction(3, 1.234, 0.0))<1e-6 and
        abs(fock_wavefunction(3, 1.234, 3.57)
            +fock_wavefunction(3, 1.234, -3.57))<1e-6 and
        abs(fock_wavefunction(4, 1.234, 3.57)
            -fock_wavefunction(4, 1.234, -3.57))<1e-6), \
        "parity check failed"

    assert (
        abs(scipy.integrate.quad(
            lambda x: abs(fock_wavefunction(1, 0.0, x))**2,
            -10.0, 10.0)[0] - 1)<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: abs(fock_wavefunction(2, 2.781, x))**2,
            -10.0, 10.0)[0] - 1)<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: abs(fock_wavefunction(3, -1.57, x))**2,
            -10.0, 10.0)[0] - 1)<1e-6), \
        "normalization check failed"

    assert (
        abs(scipy.integrate.quad(
            lambda x: (fock_wavefunction(0, 0.0, x).conjugate()
                *fock_wavefunction(2, 0.0, x)).real,
            -10.0, 10.0)[0])<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: (fock_wavefunction(0, 0.0, x).conjugate()
                *fock_wavefunction(2, 0.0, x)).imag,
            -10.0, 10.0)[0])<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: (fock_wavefunction(1, 3.456, x).conjugate()
                *fock_wavefunction(3, 2.456, x)).real,
            -10.0, 10.0)[0])<1e-6 and
        abs(scipy.integrate.quad(
            lambda x: (fock_wavefunction(1, 2.456, x).conjugate()
                *fock_wavefunction(3, 2.456, x)).imag,
            -10.0, 10.0)[0])<1e-6), \
        "orthogonality check failed"

    for n in range(10):
        assert (
            abs(scipy.integrate.quad(
                lambda x: abs(fock_wavefunction(n, 0, x))**2*x**2,
                -10.0, 10.0)[0]-(n/2+1/4))<1e-6), \
            "expectation of x^2 check failed"

def test_coherent_psi():
    dim = 20
    alpha = 2.0+1.0j
    psi = coherent_psi(dim, alpha)
    assert abs(np.linalg.norm(psi)-1)<1e-6, \
        "normalization check failed"
    assert abs(expectation(destroy(dim), psi)-alpha)<1e-4, \
        "expectation value check failed"
    assert abs(expectation(destroy(dim), psi_to_rho(psi))-alpha)<1e-4, \
        "expectation value check failed"

def test_cat_psi():
    dim = 10
    alpha = 1.0
    cat_even = cat_psi(dim, alpha, 0)
    cat_odd = cat_psi(dim, alpha, np.pi)
    eta = 0.5
    assert abs(1-np.linalg.norm(cat_odd))<1e-6, "norm test failed"
    assert abs(1-np.linalg.norm(cat_even))<1e-6, "norm test failed"
    assert abs(overlap(cat_even, cat_odd))<1e-6, "orthogonality test failed"
    assert abs(1-np.trace(psi_to_rho(cat_odd)))<1e-6, "trace test failed"
    assert abs(1-np.trace(psi_to_rho(cat_even)))<1e-6, "trace test failed"
    assert abs(1-np.trace(apply_loss(cat_odd, eta)))<1e-6, "trace test failed"
    assert abs(1-np.trace(apply_loss(cat_even, eta)))<1e-6, "trace test failed"

    for psi, sign in [(cat_even, 1), (cat_odd, -1)]: # check eigenvalues of cat
        rho = apply_loss(psi, eta)                   # DM after loss against
        evals = sorted(np.linalg.eig(rho)[0])[-2:]   # analytical calculation
        N = 0.5 / (1+sign*np.exp(-2*abs(alpha)**2))
        term = N*(np.exp(-2*(1-eta)*abs(alpha)**2)
            +sign*np.exp(-2*eta*abs(alpha)**2))
        assert abs(0.5-term-evals[0])<1e-6 and abs(0.5+term-evals[1])<1e-6, \
            "eigenvalue test failed"

    dim = 20
    alpha = 2.0
    cat = cat_psi(dim, alpha, 0, N_comps=3)
    assert sum(abs(cat[1::3])**2)+sum(abs(cat[2::3])**2)<1e-6, \
        "3-component cat population mod 3 test failed"

    a = destroy(dim)
    cat1 = cat_psi(dim, alpha, 0, N_comps=3)
    cat2 = cat_psi(dim, alpha, 2*np.pi/3, N_comps=3)
    assert np.linalg.norm(cat2 - normalize(a @ cat1))<1e-6, \
        "3-component cat a-action test failed"

def test_thermal_rho():
    dim = 40
    n = 2.0
    a = destroy(dim)
    rho = thermal_rho(dim, n)
    assert abs(expectation(a.conj().T @ a, rho)-n)<1e-4, \
        "expectation value check failed"

def test_qt_correlated_thermal_rho():
    dim1 = 16
    dim2 = 16

    nreps = 5
    for _ in range(nreps):
        n1 = np.random.random()
        n2 = np.random.random()
        cov = (np.sqrt(n1*n2)*
            np.random.random()*
            np.exp(2j*np.pi*np.random.random()))
        rho = qt_correlated_thermal_rho(dim1, dim2, n1, n2, cov)

        N1 = qt.tensor(qt.num(dim1), qt.qeye(dim2))
        N2 = qt.tensor(qt.qeye(dim1), qt.num(dim2))
        G = qt.tensor(qt.destroy(dim1).dag(), qt.destroy(dim2))
        assert abs(qt.expect(N1, rho)/n1-1)<1e-2, "correlated rho_th test failed"
        assert abs(qt.expect(N2, rho)/n2-1)<1e-2, "correlated rho_th test failed"
        assert abs(qt.expect(G, rho)/cov-1)<1e-2, "correlated rho_th test failed"


def test_displaced_fock_psi():
    dim = 30
    alpha = 2.0
    n = 2
    psi = displaced_fock_psi(dim, n, alpha)
    a = destroy(dim)

    assert abs(expectation(a, psi)-alpha)<1e-6, \
        "displaced fock state test failed"
    assert abs(expectation(a.conj().T @ a, psi)-(n+abs(alpha)**2))<1e-6, \
        "displaced fock state test failed"

def test_apply_loss():
    dim = 20
    alpha = 2.0+1.0j
    eta = 0.65
    psi = coherent_psi(dim, alpha)
    rho = apply_loss(psi, eta)
    assert abs(expectation(destroy(dim), rho)-alpha*np.sqrt(eta))<1e-4, \
        "loss check failed"

    dim = 40
    n = 2.0
    eta = 0.57
    a = destroy(dim)
    rho = apply_loss(thermal_rho(dim, n), eta)
    assert abs(expectation(a.conj().T @ a, rho)-eta*n)<1e-4, \
        "loss check failed"

if __name__ == "__main__":
    test_fock_wavefunction()
    test_coherent_psi()
    test_cat_psi()
    test_thermal_rho()
    test_qt_correlated_thermal_rho()
    test_displaced_fock_psi()
    test_apply_loss()
