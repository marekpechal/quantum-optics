import numpy as np
import qutip as qt
import scipy
import matplotlib.pyplot as plt
from quantum_optics.kiilerich import (
    get_cascaded_system_H_cops,
    solve_cascaded_system,
    get_emission_couplings,
    get_absorption_couplings,
    )
from quantum_optics.core import (
    qt_correlated_thermal_rho,
    )

def solve_cat(
        dim_therm_res = 6,
        dim_kpo = 6,
        dim_out_res = 6,
        kappa_therm_res = 1.0,
        nmean_therm_res = 0.01,
        kappa_kpo = 1.0,
        tmax = 10.0,
        num_tpts = 401,
        disc_pts = 4001
        ):

    tlist = np.linspace(0, tmax, num_tpts)
    gamma_up_therm_res = kappa_therm_res*nmean_therm_res/(nmean_therm_res+1)
    nss_kpo = 2*kappa_therm_res*nmean_therm_res/(
        kappa_therm_res/2+kappa_kpo/2-gamma_up_therm_res/2)

    def mode_shape(t):
        return kappa_kpo*np.exp(-kappa_kpo*t/2)

    def H_KPO(t):
        return qt.qzero(dim_kpo)

    # start in the steady state of the thermal resonator + KPO,
    # with the output resonator empty
    rho0_th_KPO = qt_correlated_thermal_rho(
        dim_therm_res,
        dim_kpo,
        nmean_therm_res,
        nss_kpo,
        2*nmean_therm_res*np.sqrt(kappa_therm_res*kappa_kpo)/
            (gamma_up_therm_res-kappa_therm_res-kappa_kpo))
    rho0 = qt.tensor(rho0_th_KPO, qt.fock_dm(dim_out_res, 0))

    # first simulate just the thermal resonator + KPO to find mode decomposition
    H, c_ops = get_cascaded_system_H_cops(
        [
            dim_therm_res,
            dim_kpo],
        [
            lambda t: qt.qzero(dim_therm_res),
            H_KPO],
        [
            lambda t: np.sqrt(kappa_therm_res),
            lambda t: np.sqrt(kappa_kpo)])
    corr = qt.correlation_2op_2t(H, rho0_th_KPO, tlist, tlist,
        c_ops+[qt.tensor(
            np.sqrt(gamma_up_therm_res)*qt.destroy(dim_therm_res).dag(),
            qt.qeye(dim_kpo))],
        qt.tensor(qt.qeye(dim_therm_res), qt.destroy(dim_kpo).dag()),
        qt.tensor(qt.qeye(dim_therm_res), qt.destroy(dim_kpo)))
    plt.imshow(abs(corr))
    plt.show()
    ############

    gs = (
        [lambda t: np.sqrt(kappa_therm_res)] +
        [lambda t: np.sqrt(kappa_kpo)] +
        get_absorption_couplings(
            [mode_shape],
            (0, tmax),
            Npts=disc_pts))

    c_ops = [qt.tensor(
        np.sqrt(gamma_up_therm_res)*qt.destroy(dim_therm_res).dag(),
        qt.qeye(dim_kpo),
        qt.qeye(dim_out_res))]

    sol = solve_cascaded_system(
        [
            dim_therm_res,
            dim_kpo,
            dim_out_res],
        [
            lambda t: qt.qzero(dim_therm_res),
            H_KPO,
            lambda t: qt.qzero(dim_out_res)],
        gs, rho0, tlist, c_ops=c_ops)

    N_th = qt.tensor(
        qt.num(dim_therm_res),
        qt.qeye(dim_kpo),
        qt.qeye(dim_out_res))
    N_kpo = qt.tensor(
        qt.qeye(dim_therm_res),
        qt.num(dim_kpo),
        qt.qeye(dim_out_res))
    N_out = qt.tensor(
        qt.qeye(dim_therm_res),
        qt.qeye(dim_kpo),
        qt.num(dim_out_res))

    pops_th = [qt.expect(N_th, rho) for rho in sol.states]
    pops_kpo = [qt.expect(N_kpo, rho) for rho in sol.states]
    pops_out = [qt.expect(N_out, rho) for rho in sol.states]
    plt.plot(tlist, pops_th)
    plt.plot(tlist, pops_kpo)
    plt.plot(tlist, pops_out)
    plt.plot(tlist, [nmean_therm_res for t in tlist], "--")
    plt.plot(tlist, [nss_kpo for t in tlist], "--")
    plt.grid()
    plt.show()


    return tlist, sol

if __name__ == "__main__":
    tlist, sol = solve_cat()
