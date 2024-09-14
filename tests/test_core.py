import scipy.integrate
import numpy as np
from quantum_optics.core import (
    fock_wavefunction,
    normalize,
    overlap,
    destroy,
    expectation,
    psi_to_rho,
    coherent_psi,
    cat_psi,
    thermal_rho,
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
    test_apply_loss()
