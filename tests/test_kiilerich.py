import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from quantum_optics.kiilerich import (
    solve_cascaded_system,
    get_emission_couplings,
    )

def test_cascaded_system1(plot=False):
    """Test of cascaded system simulation with 2 subsystems.

    System 1 = spontaneously decaying resonator
    System 2 = initially empty resonator with time-dependent coupling
        designed to catch the exponential mode from the spontaneous decay

    Passes if the total number of photons in the systems remains constant.
    """
    dims = [5, 5]
    Nph = 4
    kappa = 1.0
    rho0 = qt.tensor(
        qt.fock(dims[0], Nph)*qt.fock(dims[0], Nph).dag(),
        qt.fock(dims[1], 0)*qt.fock(dims[1], 0).dag()
        )
    tlist = np.linspace(0, 4.0, 401)
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]
    eps = 1e-4
    gs = [
        lambda t: np.sqrt(kappa),
        lambda t: np.sqrt(kappa)*np.exp(-kappa*t/2)/np.sqrt(1+eps-np.exp(-kappa*t))]
    sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

    N1 = qt.tensor(
        qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
        qt.qeye(dims[1]))
    N2 = qt.tensor(
        qt.qeye(dims[0]),
        qt.destroy(dims[1]).dag()*qt.destroy(dims[1]))
    pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
    pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])

    assert all(abs(pops1+pops2-Nph)<1e-3), "absorption test failed"

    if plot:
        plt.plot(tlist, pops1+pops2-Nph)
        plt.plot(tlist, pops1)
        plt.plot(tlist, pops2)
        plt.grid()
        plt.show()


def test_cascaded_system2(plot=False):
    """Test of cascaded system simulation with 2 subsystems.

    System 1 = initially populated resonator with time-dependent coupling
        designed to release a 1/cosh(kappa*t/2) mode
    System 2 = initially empty resonator with time-dependent coupling
        designed to catch the emitted mode

    Passes if the total number of photons in the systems remains constant.
    """
    dims = [5, 5]
    Nph = 4
    kappa = 1.0
    rho0 = qt.tensor(
        qt.fock(dims[0], Nph)*qt.fock(dims[0], Nph).dag(),
        qt.fock(dims[1], 0)*qt.fock(dims[1], 0).dag()
        )
    tlist = np.linspace(-10.0, 10.0, 1001)
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]
    eps = 1e-4
    # mode shape = sqrt(kappa) / 2*cosh(kappa*t/2)
    gs = [
        lambda t: 0.5*np.sqrt(kappa) / np.cosh(kappa*t/2) / np.sqrt((1-np.tanh(kappa*t/2))/2),
        lambda t: -0.5*np.sqrt(kappa) / np.cosh(kappa*t/2) / np.sqrt((1+np.tanh(kappa*t/2))/2)]
    sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

    N1 = qt.tensor(
        qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
        qt.qeye(dims[1]))
    N2 = qt.tensor(
        qt.qeye(dims[0]),
        qt.destroy(dims[1]).dag()*qt.destroy(dims[1]))
    pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
    pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])

    assert all(abs(pops1+pops2-Nph)<1e-3), "absorption test failed"

    if plot:
        plt.plot(tlist, pops1+pops2-Nph)
        plt.plot(tlist, pops1)
        plt.plot(tlist, pops2)
        plt.grid()
        plt.show()

def test_cascaded_system3(plot=False):
    """Test of cascaded system simulation with 3 subsystems.

    System 1 = spontaneously decaying resonator
    System 2 = resonator which just reflects the signal with extra phase shift
    System 3 = initially empty resonator with time-dependent coupling
        designed to catch the exponential mode from the spontaneous decay

    Passes if the total number of photons in the systems remains constant.
    """
    dims = [5, 5, 5]
    Nph = 4
    kappa = 1.0
    rho0 = qt.tensor(
        qt.fock(dims[0], Nph)*qt.fock(dims[0], Nph).dag(),
        qt.fock(dims[1], 0)*qt.fock(dims[1], 0).dag(),
        qt.fock(dims[2], 0)*qt.fock(dims[2], 0).dag()
        )
    tlist = np.linspace(0.0, 10.0, 1001)
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]
    eps = 1e-4
    gs = [
        lambda t: np.sqrt(kappa),
        lambda t: np.sqrt(kappa),
        lambda t: (1-kappa*t)*np.exp(-kappa*t/2)/np.sqrt(1+eps-np.exp(-kappa*t)*(kappa**2*t**2+1))]
    sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

    N1 = qt.tensor(
        qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
        qt.qeye(dims[1]),
        qt.qeye(dims[2]))
    N2 = qt.tensor(
        qt.qeye(dims[0]),
        qt.destroy(dims[1]).dag()*qt.destroy(dims[1]),
        qt.qeye(dims[2]))
    N3 = qt.tensor(
        qt.qeye(dims[0]),
        qt.qeye(dims[1]),
        qt.destroy(dims[2]).dag()*qt.destroy(dims[2]))
    pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
    pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])
    pops3 = np.array([qt.expect(N3, rho) for rho in sol.states])

    assert all(abs(pops1+pops2+pops3-Nph)<1e-3), "absorption test failed"

    if plot:
        plt.plot(tlist, pops1+pops2+pops3-Nph)
        plt.plot(tlist, pops1)
        plt.plot(tlist, pops2)
        plt.plot(tlist, pops3)
        plt.grid()
        plt.show()

def test_emission_couplings1(plot=False):
    """Test of emission couplings calculation with one mode.

    System 1 = initially populated resonator with time-dependent coupling
        designed to release a 1/cosh(kappa*t/2) mode. Coupling is calculated
        automatically with the tested function here.
    System 2 = initially empty resonator with time-dependent coupling
        designed to catch the emitted mode.
        Coupling is calculated manually.

    Passes if the total number of photons in the systems remains constant.
    """

    dims = [5, 5]
    Nph = 4
    rho0 = qt.tensor(
        qt.fock(dims[0], Nph)*qt.fock(dims[0], Nph).dag(),
        qt.fock(dims[1], 0)*qt.fock(dims[1], 0).dag()
        )
    tlist = np.linspace(-10.0, 10.0, 1001)
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]

    kappa = 1.0
    mode_shapes = [lambda t: np.sqrt(kappa) / (2*np.cosh(kappa*t/2))]
    gs = get_emission_couplings(mode_shapes, (tlist[0], tlist[-1])) + [
        lambda t: -0.5*np.sqrt(kappa) / np.cosh(kappa*t/2) / np.sqrt((1+np.tanh(kappa*t/2))/2)
        ]

    sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

    N1 = qt.tensor(
        qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
        qt.qeye(dims[1]))
    N2 = qt.tensor(
        qt.qeye(dims[0]),
        qt.destroy(dims[1]).dag()*qt.destroy(dims[1]))
    pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
    pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])

    assert all(abs(pops1+pops2-Nph)<1e-3), "coupling calculation test failed"

    if plot:
        plt.plot(tlist, pops1+pops2-Nph)
        plt.plot(tlist, pops1)
        plt.plot(tlist, pops2)
        plt.grid()
        plt.show()


def test_emission_couplings2(plot=False):
    """Test of emission couplings calculation with two modes.

    System 1 = initially populated resonator with time-dependent coupling
        designed to release a 1/cosh(kappa*t/2) mode. Coupling is calculated
        automatically with the tested function here.
    System 2 = initially populated resonator with time-dependent coupling
        designed to release a t/cosh(kappa*t/2) mode. Coupling is calculated
        automatically with the tested function here.
    System 3 = initially empty resonator with time-dependent coupling
        designed to catch one of the emitted modes.
        Coupling is calculated manually.

    Passes if the total number of photons in the systems remains constant.
    """

    dims = [5, 5, 5]
    Nph1 = 2
    Nph2 = 1
    rho0 = qt.tensor(
        qt.fock(dims[0], Nph1)*qt.fock(dims[0], Nph1).dag(),
        qt.fock(dims[1], Nph2)*qt.fock(dims[1], Nph2).dag(),
        qt.fock(dims[2], 0)*qt.fock(dims[2], 0).dag()
        )
    tlist = np.linspace(-30.0, 30.0, 2001)
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]

    kappa = 1.0
    mode_shapes = [
        lambda t: np.sqrt(3*kappa**3)*t / (2*np.pi*np.cosh(kappa*t/2)),
        lambda t: np.sqrt(kappa) / (2*np.cosh(kappa*t/2))
        ]
    gs_e = get_emission_couplings(mode_shapes, (tlist[0], tlist[-1]), Npts=40001)

    gs = gs_e + [
        lambda t: -0.5*np.sqrt(kappa) / np.cosh(kappa*t/2) / np.sqrt((1+np.tanh(kappa*t/2))/2)
        ]

    sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

    N1 = qt.tensor(
        qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
        qt.qeye(dims[1]),
        qt.qeye(dims[2]))
    N2 = qt.tensor(
        qt.qeye(dims[0]),
        qt.destroy(dims[1]).dag()*qt.destroy(dims[1]),
        qt.qeye(dims[2]))
    N3 = qt.tensor(
        qt.qeye(dims[0]),
        qt.qeye(dims[1]),
        qt.destroy(dims[2]).dag()*qt.destroy(dims[2]))
    pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
    pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])
    pops3 = np.array([qt.expect(N3, rho) for rho in sol.states])

    #assert all(abs(pops1+pops2-Nph)<1e-3), "coupling calculation test failed"

    if plot:
        plt.plot(tlist, pops2+pops3)
        plt.plot(tlist, pops1)
        plt.plot(tlist, pops2)
        plt.plot(tlist, pops3)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # test_cascaded_system1(plot=True)
    # test_cascaded_system2(plot=True)
    # test_cascaded_system3(plot=True)

    test_emission_couplings1(plot=True)
    test_emission_couplings2(plot=True)
