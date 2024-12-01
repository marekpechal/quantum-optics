"""Quantum interactions with pulses of radiation

Based on Kiilerich 2020 (arXiv:2003.04573v1 [quant-ph]).
Implementation with an arbitrary number of input and output resonators.
"""

import qutip as qt
import numpy as np
import scipy
import matplotlib.pyplot as plt

def solve_cascaded_system(dims, Hs, gs, *args, **kwargs):
    """Solve evolution of a number of cascaded systems.

       |
       v   a0(t)
       |
      [ ]--<>-- [system 1] A1
       |
       v   a1(t)
       |
      [ ]--<>-- [system 2] A2
       |
      ...
       |
      [ ]--<>-- [system m] Am
       |
       v   am(t)
       |
      [ ]--<>-- [system m+1] A{m+1}
       |
      ...

    Input-output relations:
    a1(t) = a0(t) + g*1(t) A1(t)
    a2(t) = a1(t) + g*2(t) A2(t)
    ...

    System operator evolution:
    Ak'(t)
      = ... [unitary evolution] - |gk(t)|^2 Ak(t) / 2
        - gk(t) a{k-1}(t)
      = ... [unitary evolution] - |gk(t)|^2 Ak(t) / 2
        - gk(t) (g*{k-1}(t) A{k-1}(t) + g*{k-2}(t) A{k-2}(t) + ... + a0(t))

    Re-write in matrix form (M is a matrix, G a vector):

    A'(t) = ... [unitary evolution] + M(t) A(t) - G(t) a0(t)
    where
    M(t) = [
      [-|g1(t)|^2/2, 0, 0, ...],
      [-g2(t) g*1(t), -|g2(t)|^2/2, 0, 0, ...],
      [-g3(t) g*1(t), -g3(t) g*2(t), -|g3(t)|^2/2, 0, 0, ...],
      ...
      ]
    and
    G(t) = [g1(t), g2(t), ...]

    Find master equation which results in the Heisenberg equations above:

    rho'(t) = -i[H0(t), rho(t)] [unitary evolution]
      -i[Hc(t), rho(t)]
      -D_X(t)[rho(t)]

    where
    X(t) = g*1(t) A1(t) + g*2(t) A2(t) - ...
    Hc(t) = -i sum_{i>j} (gi(t) g*j(t) Adi Aj - H.c.) / 2


    Args:
      dims (list): Dimensions of the individual systems.
      Hs (list): Hamiltonians of the individual systems
          as functions of time.
      gs (list): In/out couplings of the individual systems
          as functions of time.
      args (tuple): Additional arguments to be passed to qutip.mesolve.
      kwargs (dict): Additional arguments to be passed to qutip.mesolve.

    Returns:
      qutip.Result
    """
    A = [
        qt.tensor(*[
            (qt.destroy(dimi) if j==i else qt.qeye(dimj))
            for j, dimj in enumerate(dims)])
        for i, dimi in enumerate(dims)]

    def H(t, args):
        result = qt.tensor(*[qt.qzero(dim) for dim in dims])
        for i, gi in enumerate(gs):
            for j, gj in enumerate(gs[:i]):
                X = gi(t)*gj(t).conjugate()*A[i].dag()*A[j]
                result += -0.5j*(X - X.dag())
        for idx, Hf in enumerate(Hs):
            result += qt.tensor(*[(Hf(t) if j==idx else qt.qeye(dim))
                for j, dim in enumerate(dims)])
        return result

    def c_op(t, args):
        result = qt.tensor(*[qt.qzero(dim) for dim in dims])
        for i, gi in enumerate(gs):
            result += gi(t).conjugate()*A[i]
        return result

    if "c_ops" in kwargs:
        kwargs["c_ops"].append(c_op)
    else:
        kwargs["c_ops"] = [c_op]
    return qt.mesolve(H, *args, **kwargs)

def get_emission_couplings(mode_shapes, t_range, Npts=1001):
    """Calculate coupling to emit multiple modes in cascaded systems.

    Args:
      mode_shapes (list): Mode shapes as functions of time.
      t_range (tuple): Initial and final time (t1, t2)
      Npts (int, optional): Number of discretization points.

    Returns:
      list: List of functions of time, evaluating the couplings.
    """

    t_arr = np.linspace(*t_range, Npts)
    dt = t_arr[1]-t_arr[0]
    mode_shapes_samp = np.array([[f(t) for t in t_arr] for f in mode_shapes])

    res = []
    for i in range(len(mode_shapes_samp)):
        mode = mode_shapes_samp[len(mode_shapes_samp)-i-1]
        g = np.zeros_like(mode)
        cs = np.cumsum(abs(mode[::-1])**2)[::-1]*dt
        mask = (cs>0)
        g[mask] = mode[mask].conjugate()/np.sqrt(cs[mask])
        res.append(g)

        # compensate for distortion of other modes
        fac = np.exp(np.cumsum(-abs(res[-1])**2/2)*dt)
        for j in range(i+1, len(mode_shapes_samp)):
            mode2 = mode_shapes_samp[len(mode_shapes_samp)-j-1]

            A = np.cumsum(-res[-1]*mode2*fac)*dt
            A -= A[-1]
            A = A/fac
            mode_shapes_samp[len(mode_shapes_samp)-j-1] = (
                mode2-
                res[-1].conjugate()*A
                )

    return [scipy.interpolate.interp1d(t_arr, pts,
            bounds_error=False,
            fill_value=0.0)
        for pts in res[::-1]]

def get_absorption_couplings(mode_shapes, t_range, Npts=1001):
    """Calculate coupling to absorb multiple modes in cascaded systems.

    Args:
      mode_shapes (list): Mode shapes as functions of time.
      t_range (tuple): Initial and final time (t1, t2)
      Npts (int, optional): Number of discretization points.

    Returns:
      list: List of functions of time, evaluating the couplings.
    """

    t_arr = np.linspace(*t_range, Npts)
    dt = t_arr[1]-t_arr[0]
    mode_shapes_samp = np.array([[f(t) for t in t_arr] for f in mode_shapes])

    res = []
    for i in range(len(mode_shapes_samp)):
        mode = mode_shapes_samp[i]
        g = np.zeros_like(mode)
        cs = np.cumsum(abs(mode)**2)*dt
        mask = (cs>0)
        g[mask] = -mode[mask].conjugate()/np.sqrt(cs[mask])
        res.append(g)

        # compensate for distortion of other modes
        fac = np.exp(np.cumsum(abs(res[-1])**2/2)*dt)
        for j in range(i+1, len(mode_shapes_samp)):
            mode2 = mode_shapes_samp[j]

            A = np.cumsum(-res[-1]*mode2*fac)*dt
            A = A/fac
            mode_shapes_samp[j] = (
                mode2+
                res[-1].conjugate()*A
                )

    return [scipy.interpolate.interp1d(t_arr, pts,
            bounds_error=False,
            fill_value=0.0)
        for pts in res]
