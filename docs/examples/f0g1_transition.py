import numpy as np
import qutip as qt
import scipy
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level="INFO")

def get_closest_eigenvector(H0, psi):
    """Get Hamiltonian eigenstate(s) closest to a given state or states.

    Args:
        H0 (qutip.Qobj): Hamiltonian.
        psi (qutip.Qobj or list): State or states for which the closest
            eigenstates should be found.

    Returns:
        qutip.Qobj or list: The eigenstate(s) closest to the given state(s).
    """

    evals, evecs = H0.eigenstates()
    if isinstance(psi, list):
        return [sorted(evecs, key=lambda evec: abs(evec.overlap(p))**2)[-1]
            for p in psi]
    else:
        return sorted(evecs, key=lambda evec: abs(evec.overlap(psi))**2)[-1]

def get_closest_qr_eigenvector(H0, n_q, n_r):
    """Get qudit-resonator eigenstate closest to a given state.

    Args:
        H0 (qutip.Qobj): Hamiltonian of the qudit-resonator system.
        n_q (int): Number of excitations in the qudit.
        n_r (int): Number of excitations in the resonator.

    Returns:
        qutip.Qobj: The eigenstate closest to the given state.
    """

    psi = qt.tensor(
        qt.fock(H0.dims[0][0], n_q),
        qt.fock(H0.dims[0][1], n_r)
        )
    return get_closest_eigenvector(H0, psi)

def get_eff_drive_params(H0, Wmax, npts=4001):
    """Calculate effective f0g1 drive strength and ac Stark shift.

    Based on appendix A of Zeytinoglu 2015 (arXiv:1502.03692v1 [quant-ph]).

    Args:
        H0 (Qobj): Hamiltonian of the qubit-resonator system without drive.
        Wmax (float): Upper limit of drive strength.
        npts (int, optional): Number of discretization points.

    Returns:
        tuple:
            float: Drive frequency shift needed to make the drive resonant
                with the f0 <-> g1 transition.
            callable: Interpolation function from drive strength W to effective
                coupling g-tilde.
            callable: Interpolation function from effective coupling g-tilde to
                drive strength W.
            callable: Interpolation function from drive strength W to Stark
                shift.
            callable: Interpolation function from Stark shift to drive
                strength W.
    """

    logging.info("calculating effective drive strength and ac Stark shift")

    dim_q, dim_r = H0.dims[0]
    a_q = qt.tensor(qt.destroy(dim_q), qt.qeye(dim_r))
    a_r = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))

    dW = Wmax/(npts-1)
    Wrng = np.arange(npts)*dW

    dH_dW = 0.5*(a_q+a_q.dag())
    dH_ddet = -(a_r.dag()*a_r + a_q.dag()*a_q)

    psi_f0 = get_closest_qr_eigenvector(H0, 2, 0)
    psi_g1 = get_closest_qr_eigenvector(H0, 0, 1)

    # initialize drive frequency to be resonant with dressed transition
    dE = qt.expect(H0, psi_f0) - qt.expect(H0, psi_g1)
    H0s = H0 - dE*(a_r.dag()*a_r + a_q.dag()*a_q)
    W = 0.0
    det = 0.0
    Phis = [(psi_f0+psi_g1)/np.sqrt(2), (psi_f0-psi_g1)/np.sqrt(2)]

    g_tildes = [0.0]
    dets = [0.0]
    for _ in range(npts-1):
        A = Phis[1].dag()*dH_dW*Phis[0]
        B = Phis[1].dag()*dH_ddet*Phis[0]
        det += (-A/B)*dW
        W += dW
        H = H0s+det*dH_ddet+W*dH_dW
        Phis = get_closest_eigenvector(H, Phis)

        dets.append(det)
        g_tildes.append((qt.expect(H, Phis[1]) - qt.expect(H, Phis[0]))/2)

    return (
        dE,
        scipy.interpolate.interp1d(Wrng, g_tildes),
        scipy.interpolate.interp1d(g_tildes, Wrng),
        scipy.interpolate.interp1d(Wrng, dets),
        scipy.interpolate.interp1d(dets, Wrng),
        )

if __name__ == "__main__":
    dim_q = 4
    dim_r = 5
    alpha = -2*np.pi*200.0e6
    g = 2*np.pi*100.0e6
    delta_rq = 2*np.pi*1000.0e6
    # delta_rq = w_r - w_q
    # w_d = 2*w_q + alpha - w_r
    # => delta_q = w_q - w_d = w_r - w_q - alpha = delta_rq - alpha
    # => delta_r = w_r - w_d = 2*(w_r - w_q) - alpha = 2*delta_rq - alpha
    delta_q = delta_rq - alpha
    delta_r = 2*delta_rq - alpha


    a_q = qt.tensor(qt.destroy(dim_q), qt.qeye(dim_r))
    a_r = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))

    H0_q = delta_q*a_q.dag()*a_q + 0.5*alpha*a_q.dag()*a_q.dag()*a_q*a_q
    H0_r = delta_r*a_r.dag()*a_r
    H0 = H0_q + H0_r + g*(a_q.dag()*a_r + a_r.dag()*a_q)

    (dE,
        W_to_gtilde,
        gtilde_to_W,
        W_to_starkshift,
        starkshift_to_W) = get_eff_drive_params(H0, 2*np.pi*200.0e6)

    psi_f0 = get_closest_qr_eigenvector(H0, 2, 0)
    psi_g1 = get_closest_qr_eigenvector(H0, 0, 1)

    # correct drive frequency to account for dressing by JC coupling
    H0 -= dE*(a_r.dag()*a_r + a_q.dag()*a_q)

    T = 0.25e-6
    tlist = np.linspace(0, T, 1001)
    gt0 = np.pi/T

    Wlist = [gtilde_to_W(gt0*np.sin(np.pi*t/T)**2) for t in tlist]
    detlist = [W_to_starkshift(W) for W in Wlist]
    phaselist = np.cumsum(detlist)*(tlist[1]-tlist[0])
    Wcarr = np.array(Wlist)*np.exp(-1j*np.array(phaselist))
    W = scipy.interpolate.interp1d(tlist, Wcarr,
        bounds_error=False, fill_value=0)

    logging.info("simulating evolution with stark shift compensation")

    def H(t, args):
        return H0 + (a_q.dag()*W(t) + a_q*W(t).conjugate())/2
    sol = qt.sesolve(H, psi_f0, tlist)
    pops = [abs(psi_f0.overlap(psi))**2 for psi in sol.states]
    plt.plot(tlist/1e-9, pops, label="with Stark shift compensation")

    logging.info("simulating evolution without stark shift compensation")

    def H(t, args):
        return H0 + abs(W(t))*(a_q.dag() + a_q)/2
    sol = qt.sesolve(H, psi_f0, tlist)
    pops = [abs(psi_f0.overlap(psi))**2 for psi in sol.states]
    plt.plot(tlist/1e-9, pops, label="without Stark shift compensation")

    plt.xlabel("Time [ns]")
    plt.ylabel("$|f0\\rangle$ population")
    plt.legend()
    plt.grid()
    plt.show()
