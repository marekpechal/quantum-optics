from quantum_optics.core import (
    cat_psi,
    fock_wavefunction,
    expectation,
    )

from quantum_optics.measurements import (
    draw_from_pdf,
    wigner_W,
    )

from quantum_optics.lvovsky import (
    iteration_step,
    )

import matplotlib.pyplot as plt
import numpy as np
import time
import logging
logging.basicConfig(level="INFO")

if __name__ == "__main__":
    dim = 16                         # Hilbert space dimension
    alpha = 2.0                      # Cat size
    eta = 0.8                        # Quantum efficiency
    psi = cat_psi(dim, alpha, np.pi) # Odd cat state vector in Fock basis

    theta_rng = np.linspace(0, np.pi, 17)[:-1]  # Quadrature angles
    reps = 200                                  # Number of measurements
                                                #     per quadrature

    theta_arr = np.array([theta                 # Array of quadrature angles
        for theta in theta_rng                  #     with repetitions
        for _ in range(reps)])

    xmax = 8.0        # Range for pdf discretization
    n_subdiv = 1000   # Subdivisions for pdf discretization

    x_arr = []
    for idx, theta in enumerate(theta_rng):
        logging.info(f"generating {reps} samples for quadrature " \
            f"{idx+1}/{len(theta_rng)}")
        t0 = time.time()
        pdf = lambda x: abs(sum([a*fock_wavefunction(n, theta, x)
            for n, a in enumerate(psi)]))**2  # state -> wave function -> pdf
        samples = draw_from_pdf(pdf, reps, (-xmax, xmax), n_subdiv, seed=idx)
        x_arr = np.concatenate((x_arr, samples))
        logging.info(f"done in {time.time()-t0:.2f} seconds")
    x_arr = (np.sqrt(eta)*x_arr
        + np.sqrt(1-eta)*0.5*np.random.normal(size=len(x_arr)))

    rho = np.eye(dim).astype(complex)
    n_iters = 200
    plt.figure(figsize = (6, 6))
    plt.xlabel("Re $\\alpha$")
    plt.ylabel("Im $\\alpha$")
    for idx in range(n_iters):  # Iterate Lvovsky algorithm
        logging.info(f"ML iteration {idx+1}/{n_iters}")
        t0 = time.time()
        rho = iteration_step(rho, theta_arr, x_arr, eta=eta)
        logging.info(f"done in {time.time()-t0:.2f} seconds")
        logging.info(f"fidelity = {expectation(rho, psi).real:.3f}")

        t0 = time.time()
        logging.info(f"drawing plot of Wigner function")
        xrng = np.linspace(-3.0, 3.0, 101)
        wvals = np.array([[wigner_W(rho, x+1j*y) for x in xrng] for y in xrng])
        plt.clf()
        plt.imshow(wvals[::-1], vmin=-1, vmax=1, cmap="RdBu")
        plt.savefig(f"wigner_plot_{idx+1}.png")
        logging.info(f"done in {time.time()-t0:.2f} seconds")
