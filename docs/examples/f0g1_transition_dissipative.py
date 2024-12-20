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

def make_proj_2d(u1, u2):
    """Make projector onto space spanned by two vectors.

    The vectors do not need to be orthogonal or normalized.

    Args:
        u1 (Qobj): 1st vector.
        u2 (Qobj): 2nd vector.

    Returns:
        Qobj: Projector operator.
    """

    u1 = u1/np.sqrt(u1.dag()*u1)
    u2 = u2 - u1*(u1.dag()*u2)
    u2 = u2/np.sqrt((u2.dag()*u2).real)
    return u1*u1.dag()+u2*u2.dag()


def get_eff_drive_params(H0, Wmax, npts=10001, ops=None):
    """TBD

    Based on appendix A of Zeytinoglu 2015 (arXiv:1502.03692v1 [quant-ph]).

    Args:
        H0 (Qobj): Hamiltonian of the qubit-resonator system without drive.
        Wmax (float): Upper limit of drive strength.
        npts (int, optional): Number of discretization points.
        ops (list or None, optional): Additional operators to restrict to
            the two-level subspace.

    Returns:
        TBD
    """

    logging.info("calculating effective drive strength and ac Stark shift")

    dim_q, dim_r = H0.dims[0]
    a_q = qt.tensor(qt.destroy(dim_q), qt.qeye(dim_r))
    a_r = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))


    dW = Wmax/(npts-1)
    Wrng = np.arange(npts)*dW

    dH_dW = 0.5*(a_q+a_q.dag())
    dH_ddet = -(a_r.dag()*a_r + a_q.dag()*a_q)
    I = qt.tensor(qt.qeye(dim_q), qt.qeye(dim_r))

    psi_f0 = get_closest_qr_eigenvector(H0, 2, 0)
    psi_g1 = get_closest_qr_eigenvector(H0, 0, 1)
    # Notes: returned states are normalized and orthogonal
    # (because they are in different excitation number eigenspaces)

    # initialize drive frequency to be resonant with dressed transition
    dE = (qt.expect(H0, psi_f0) - qt.expect(H0, psi_g1)).real
    H0s = H0 - dE*(a_r.dag()*a_r + a_q.dag()*a_q)
    E0 = qt.expect(H0s, psi_f0).real
    H0s = H0s - E0*I

    W = 0.0
    det = 0.0
    dE0 = 0.0
    Phis = [psi_f0, psi_g1]
    basis = [psi_f0, psi_g1]
    bases = [basis]
    H_restr = [[u.dag()*H0s*v for v in basis] for u in basis]

    H_restr_lst = [H_restr]
    dets = [0.0]
    for _ in range(npts-1):
        # update drive strength
        W += dW

        H = H0s+det*dH_ddet+W*dH_dW-dE0*I
        Phis_new = get_closest_eigenvector(H, Phis)
        proj = make_proj_2d(*Phis_new)
        basis = [proj*u for u in basis]
        H_restr = [[u.dag()*H*v for v in basis] for u in basis]

        # update ac Stark shift and global energy shift
        exp0 = qt.expect(dH_ddet, basis[0])
        exp1 = qt.expect(dH_ddet, basis[1])
        ddet = (H_restr[0][0]-H_restr[1][1]).real/(exp1-exp0)
        dE0 += (H_restr[0][0]+ddet*exp0).real
        det += ddet
        dets.append(det)

        H = H0s+det*dH_ddet+W*dH_dW-dE0*I
        Phis_new = get_closest_eigenvector(H, Phis)
        proj = make_proj_2d(*Phis_new)
        basis = [proj*u for u in basis]
        bases.append(basis)
        H_restr = [[u.dag()*H*v for v in basis] for u in basis]

        H_restr_lst.append(H_restr)
        Phis = Phis_new

    dets = np.array(dets)
    bases = np.array(bases)

    if ops is None:
        ops = []
    ops = np.array(
        [[[[u.dag()*op*v for v in basis] for u in basis]
            for basis in bases]
                for op in ops])

    H_restr_lst = np.array(H_restr_lst)
    for c, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        plt.subplot(2, 2, c+1)
        plt.plot(H_restr_lst[:, i, j].real)
        plt.plot(H_restr_lst[:, i, j].imag)
        plt.grid()
    plt.show()

    ops_f = []
    for op in ops:
        W_to_op11 = scipy.interpolate.interp1d(Wrng, op[:, 0, 0])
        W_to_op12 = scipy.interpolate.interp1d(Wrng, op[:, 0, 1])
        W_to_op21 = scipy.interpolate.interp1d(Wrng, op[:, 1, 0])
        W_to_op22 = scipy.interpolate.interp1d(Wrng, op[:, 1, 1])
        def W_to_op(W):
            op11 = W_to_op11(W)
            op12 = W_to_op12(W)
            op21 = W_to_op21(W)
            op22 = W_to_op22(W)
            return np.array([
                [op11, op12],
                [op21, op22]
                ])
        ops_f.append(W_to_op)


    W_to_Heff11 = scipy.interpolate.interp1d(Wrng, H_restr_lst[:, 0, 0])
    W_to_Heff12 = scipy.interpolate.interp1d(Wrng, H_restr_lst[:, 0, 1])
    W_to_Heff21 = scipy.interpolate.interp1d(Wrng, H_restr_lst[:, 1, 0])
    W_to_Heff22 = scipy.interpolate.interp1d(Wrng, H_restr_lst[:, 1, 1])
    def W_to_Heff(W):
        Heff11 = W_to_Heff11(W)
        Heff12 = W_to_Heff12(W)
        Heff21 = W_to_Heff21(W)
        Heff22 = W_to_Heff22(W)
        return np.array([
            [Heff11, Heff12],
            [Heff21, Heff22]
            ])

    return (
        dE,
        scipy.interpolate.interp1d(Wrng, dets),
        (Wrng, bases),
        W_to_Heff,
        ops_f,
        )

if __name__ == "__main__":
    dim_q = 4
    dim_r = 5
    alpha = -2*np.pi*200.0e6
    g = 2*np.pi*100.0e6
    delta_rq = 2*np.pi*1000.0e6
    kappa = 2*np.pi*1.5e6
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
    psi_g1_bare = qt.tensor(qt.fock(dim_q, 0), qt.fock(dim_r, 1))
    psi_g0_bare = qt.tensor(qt.fock(dim_q, 0), qt.fock(dim_r, 0))
    H_loss = kappa*a_r.dag()*a_r

    H0_diss = H0-0.5j*H_loss

    psi_f0 = get_closest_qr_eigenvector(H0_diss, 2, 0)
    psi_g1 = get_closest_qr_eigenvector(H0_diss, 0, 1)
    psi_g0 = get_closest_qr_eigenvector(H0_diss, 0, 0)

    proj_g1 = psi_g1*psi_g1.dag()


    (dE,
        W_to_det,
        (Wrng, bases),
        W_to_Heff,
        (W_to_proj_g1, ),
        ) =  get_eff_drive_params(H0_diss, 2*np.pi*400.0e6, ops=[proj_g1])


    # test by comparing two-level model with full JC + dissipation
    W0 = 2*np.pi*100.0e6
    T = 1e-6
    tlist = np.linspace(0, T, 1001)

    def W(t):
        return W0*np.sin(np.pi*t/T)**2

    # correct drive frequency to account for dressing by JC coupling
    def H(t, args):
        return (H0_diss -
            (dE+W_to_det(W(t)))*(a_r.dag()*a_r + a_q.dag()*a_q) +
            (a_q.dag()*W(t) + a_q*W(t).conjugate())/2)

    sol = qt.sesolve(H, psi_f0, tlist, options={"normalize_output":False}) #options=options)

    emission = np.array([qt.expect(proj_g1, state)
        for state in sol.states])

    plt.plot(tlist, emission)

    def Heff(t):
        return W_to_Heff(W(t))

    def gen(t, psi):
        return -1j*Heff(t) @ psi
    sol2 = scipy.integrate.solve_ivp(gen, (0, T),
        np.array([1., 0.], dtype=complex),
        t_eval=tlist)

    Y = [(psi.conjugate() @ (W_to_proj_g1(W(t)) @ psi)).real
        for t, psi in zip(tlist, sol2.y.transpose())]

    plt.plot(tlist, Y, "--")

    plt.grid()
    plt.show()
