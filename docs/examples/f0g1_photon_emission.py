import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def get_closest_eigenvector(H0, psi):
    evals, evecs = H0.eigenstates()
    return sorted(evecs, key=lambda evec: abs(evec.overlap(psi))**2)[-1]

def get_closest_qr_eigenvectors(H0, n_q, n_r):
    psi = qt.tensor(
        qt.fock(H0.dims[0][0], n_q),
        qt.fock(H0.dims[0][1], n_r)
        )
    return get_closest_eigenvector(H0, psi)

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

    psi_f0 = get_closest_qr_eigenvectors(H0, 2, 0)
    psi_g1 = get_closest_qr_eigenvectors(H0, 0, 1)

    # correct drive frequency to account for dressing by JC coupling
    dE = qt.expect(H0, psi_f0) - qt.expect(H0, psi_g1)
    H0 -= dE*(a_r.dag()*a_r + a_q.dag()*a_q)

    W0 = 2*np.pi*10.0e6
    g_tilde = W0*g*alpha/(np.sqrt(2)*delta_rq*(delta_rq-alpha))

    T = np.pi/abs(g_tilde)
    tlist = np.linspace(0, T, 1001)
    def H(t, args):
        return H0 + W0*np.sin(np.pi*t/T)**2*(a_q.dag() + a_q)/2
    sol = qt.sesolve(H, psi_f0, tlist)
    pops = [abs(psi_f0.overlap(psi))**2 for psi in sol.states]
    plt.plot(tlist, pops)
    plt.grid()
    plt.show()
