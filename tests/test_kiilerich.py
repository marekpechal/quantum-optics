import numpy as np
import qutip as qt
import scipy
import matplotlib.pyplot as plt
from quantum_optics.kiilerich import (
    solve_cascaded_system,
    get_emission_couplings,
    get_absorption_couplings,
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
        Coupling is calculated automatically.

    Passes if the total number of photons in the output resonator at the
    end of the simulation equals the number of photons in the input resonator
    with the matching mode at the beginning.
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
    gs_e = get_emission_couplings(mode_shapes,
        (tlist[0], tlist[-1]),
        Npts=40001)

    for m_idx in range(2): # simulate for each input mode matched to the output
        gs_a = get_absorption_couplings([mode_shapes[m_idx]],
            (tlist[0], tlist[-1]),
            Npts=40001)

        gs = gs_e + gs_a

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

        pops_m = [pops1, pops2][m_idx]
        assert abs(pops3[-1]-pops_m[0])<1e-3, \
            "coupling calculation test failed"

        if plot:
            plt.plot(tlist, pops1)
            plt.plot(tlist, pops2)
            plt.plot(tlist, pops3)
            plt.grid()
            plt.show()


def test_emission_absorption_couplings1(plot=False):
    """Test of emission & absorption couplings calculation with 1+1 mode.

    System 1 = initially populated resonator with time-dependent coupling
        designed to release a 1/cosh(kappa*t/2) mode. Coupling is calculated
        automatically with the tested function here.
    System 2 = initially empty resonator with time-dependent coupling
        designed to catch the emitted mode.
        Coupling is calculated automatically with the tested function here.

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
    gs = (
        get_emission_couplings(mode_shapes, (tlist[0], tlist[-1])) +
        get_absorption_couplings(mode_shapes, (tlist[0], tlist[-1]))
        )

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

def test_emission_absorption_couplings2(plot=False):
    """Test of emission & absorption couplings calculation with 2+2 modes.

    System 1 = initially populated resonator with time-dependent coupling
        designed to release a 1/cosh(kappa*t/2) mode. Coupling is calculated
        automatically with the tested function here.
    System 2 = initially populated resonator with time-dependent coupling
        designed to release a t/cosh(kappa*t/2) mode. Coupling is calculated
        automatically with the tested function here.
    System 3 = initially empty resonator with time-dependent coupling
        designed to catch one of the emitted modes.
        Coupling is calculated automatically.
    System 4 = initially empty resonator with time-dependent coupling
        designed to catch the other emitted mode.
        Coupling is calculated automatically.

    Passes if the total numbers of photons in the output resonators at the
    end of the simulation equal the numbers of photons in the input resonators
    with the matching modes at the beginning.
    """

    dims = [4, 4, 4, 4]
    Nph1 = 1
    Nph2 = 2
    rho0 = qt.tensor(
        qt.fock(dims[0], Nph1)*qt.fock(dims[0], Nph1).dag(),
        qt.fock(dims[1], Nph2)*qt.fock(dims[1], Nph2).dag(),
        qt.fock(dims[2], 0)*qt.fock(dims[2], 0).dag(),
        qt.fock(dims[3], 0)*qt.fock(dims[3], 0).dag()
        )
    tlist = np.linspace(-30.0, 30.0, 2001)
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]

    kappa = 1.0
    ph1 = np.exp(1.23j)
    ph2 = np.exp(2.72j)
    mode_shapes = [
        lambda t: np.sqrt(3*kappa**3)*t*ph1 / (2*np.pi*np.cosh(kappa*t/2)),
        lambda t: np.sqrt(kappa)*ph2 / (2*np.cosh(kappa*t/2)),
        ]

    # test it for different orderings of the input and output resonators
    for m_idx in range(4):
        gs = (
            get_emission_couplings(
                mode_shapes if (m_idx % 2 == 0) else mode_shapes[::-1],
                (tlist[0], tlist[-1]),
                Npts=40001) +
            get_absorption_couplings(
                mode_shapes if ((m_idx//2) % 2 == 0) else mode_shapes[::-1],
                (tlist[0], tlist[-1]),
                Npts=40001)
            )

        sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

        N1 = qt.tensor(
            qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
            qt.qeye(dims[1]),
            qt.qeye(dims[2]),
            qt.qeye(dims[3]))
        N2 = qt.tensor(
            qt.qeye(dims[0]),
            qt.destroy(dims[1]).dag()*qt.destroy(dims[1]),
            qt.qeye(dims[2]),
            qt.qeye(dims[3]))
        N3 = qt.tensor(
            qt.qeye(dims[0]),
            qt.qeye(dims[1]),
            qt.destroy(dims[2]).dag()*qt.destroy(dims[2]),
            qt.qeye(dims[3]))
        N4 = qt.tensor(
            qt.qeye(dims[0]),
            qt.qeye(dims[1]),
            qt.qeye(dims[2]),
            qt.destroy(dims[3]).dag()*qt.destroy(dims[3]))
        pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
        pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])
        pops3 = np.array([qt.expect(N3, rho) for rho in sol.states])
        pops4 = np.array([qt.expect(N4, rho) for rho in sol.states])

        if ((m_idx % 2) == ((m_idx//2) % 2)):
            assert abs(pops1[0]-pops3[-1])+abs(pops2[0]-pops4[-1])<1e-3, \
                "coupling calculation test failed"
        else:
            assert abs(pops1[0]-pops4[-1])+abs(pops2[0]-pops3[-1])<1e-3, \
                "coupling calculation test failed"

        if plot:
            plt.plot(tlist, pops1)
            plt.plot(tlist, pops2)
            plt.plot(tlist, pops3)
            plt.plot(tlist, pops4)
            plt.grid()
            plt.show()

def test_emission_absorption_couplings3(plot=False):
    """Test of emission & absorption couplings calculation with 1+1 mode.

    System 1 = initially populated resonator with time-dependent coupling
        designed to release a randomly generated mode.
    System 2 = initially empty resonator with time-dependent coupling
        designed to catch the emitted mode.

    Passes if the final state of system 2 is close enough to the initial
    state of system 1.
    """

    dims = [5, 5]
    tlist = np.linspace(-20.0, 20.0, 2001)
    dt = tlist[1]-tlist[0]
    Hs = [(lambda t, dim=dim: qt.qzero(dim)) for dim in dims]

    kappa = 1.0
    order = 3

    Nreps = 5

    for _ in range(Nreps):
        psi_vec = np.random.normal(size=dims[0])+1j*np.random.normal(size=dims[0])
        psi_vec = psi_vec/np.linalg.norm(psi_vec) # normalize
        psi = qt.Qobj(psi_vec)
        rho0 = qt.tensor(
            psi*psi.dag(),
            qt.fock(dims[1], 0)*qt.fock(dims[1], 0).dag()
            )

        c = np.random.normal(size=order)+1j*np.random.normal(size=order)
        mode = lambda t: sum([a*t**i for i,a in enumerate(c)])/np.cosh(kappa*t/2)
        N = 1/np.sqrt(sum([abs(mode(t))**2 for t in tlist])*dt)
        mode_norm = lambda t: N*mode(t)

        mode_shapes = [mode_norm]
        gs = (
            get_emission_couplings(mode_shapes, (tlist[0], tlist[-1])) +
            get_absorption_couplings(mode_shapes, (tlist[0], tlist[-1]))
            )

        sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

        N1 = qt.tensor(
            qt.destroy(dims[0]).dag()*qt.destroy(dims[0]),
            qt.qeye(dims[1]))
        N2 = qt.tensor(
            qt.qeye(dims[0]),
            qt.destroy(dims[1]).dag()*qt.destroy(dims[1]))
        pops1 = np.array([qt.expect(N1, rho) for rho in sol.states])
        pops2 = np.array([qt.expect(N2, rho) for rho in sol.states])

        F = qt.fidelity(sol.states[0].permute([1, 0]), sol.states[-1])
        assert F>0.999, "coupling calculation test failed"

        if plot:
            plt.plot(tlist, pops1+pops2)
            plt.plot(tlist, pops1)
            plt.plot(tlist, pops2)
            plt.grid()
            plt.show()

def test_absorption_couplings1(plot=False):
    """Test of absorption couplings calculation.

    Simulate emission from a driven two-level system. Then find shapes of
    most populated modes by SVD of the ad.a correlation function and
    set up cascaded system to absorb the content of those modes into
    resonators.

    Passes if final mean numbers of photons in the output resonators match
    the populations of the modes calculated from the correlation function.
    """

    dim_sys = 2
    sigma = 0.5
    kappa = 1.0
    T = 5.0
    a = qt.destroy(dim_sys)
    H_sys = lambda t: (np.sqrt(np.pi/2)/sigma)*np.exp(-t**2/(2*sigma**2))*qt.sigmax()/2
    rho0 = qt.fock(dim_sys, 0)*qt.fock(dim_sys, 0).dag()
    tlist = np.linspace(-T/2, 3*T/2, 1001)
    dt = tlist[1]-tlist[0]
    c_ops = [np.sqrt(kappa)*a]


    sol = qt.mesolve(H_sys, rho0, tlist,
        c_ops=c_ops)

    pops = [qt.expect(rho0, rho).real for rho in sol.states]

    corr = qt.correlation.correlation_2op_2t(H_sys, rho0, tlist, tlist-tlist[0], c_ops, a.dag(), a)
    for i in range(corr.shape[0]):
        corr[i, i:] = corr[i, :corr.shape[1]-i]
    for i in range(corr.shape[0]):
        corr[i:, i] = corr[i, i:].conjugate()

    U, S, V = np.linalg.svd(corr*dt)

    Nmodes = 3
    dim_res = 4
    modes = [scipy.interpolate.interp1d(tlist, U[:, i]/np.sqrt(dt),
            bounds_error=False, fill_value=0.0)
        for i in range(Nmodes)]
    gs = [lambda t: np.sqrt(kappa)] + get_absorption_couplings(modes, (tlist[0], tlist[-1]))
    dims = [dim_sys] + [dim_res]*Nmodes
    Hs = [H_sys] + [lambda t: qt.qzero(dim_res)]*Nmodes
    rho0 = qt.tensor(*[qt.fock(dim, 0)*qt.fock(dim, 0).dag() for dim in dims])
    sol = solve_cascaded_system(dims, Hs, gs, rho0, tlist)

    Ns = [qt.expect(qt.tensor(*([qt.qeye(dim_sys)]+[
        (qt.destroy(dim_res).dag()*qt.destroy(dim_res)
            if i==j else qt.qeye(dim_res))
        for j in range(Nmodes)])), sol.states[-1])
            for i in range(Nmodes)]

    assert all(abs(Ns/S[:Nmodes]-1)<1e-2), "coupling calculation test failed"

    if plot:
        plt.subplot(3, 1, 1)
        plt.plot(tlist, pops)
        plt.subplot(3, 1, 2)
        for i in range(3):
            plt.plot(tlist, abs(U[:, i]))
        plt.subplot(3, 1, 3)
        plt.imshow(abs(corr))
        plt.show()


if __name__ == "__main__":
    test_cascaded_system1(plot=True)
    test_cascaded_system2(plot=True)
    test_cascaded_system3(plot=True)

    test_emission_couplings1(plot=True)
    test_emission_couplings2(plot=True)

    test_absorption_couplings1(plot=True)

    test_emission_absorption_couplings1(plot=True)
    test_emission_absorption_couplings2(plot=True)
    test_emission_absorption_couplings3(plot=True)
