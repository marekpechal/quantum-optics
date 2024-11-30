"""Quantum interactions with pulses of radiation

Based on Kiilerich 2020 (arXiv:2003.04573v1 [quant-ph]).
Implementation with an arbitrary number of input and output resonators.

1. A number of cascaded systems
(we do not distinguish between the input and output resonators and the
quantum system for now)

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

Find form of master equation which results in the Heisenberg equations above:

rho'(t) = -i[H0(t), rho(t)] [unitary evolution]
  -i[Hc(t), rho(t)]
  -D_X(t)[rho(t)]

where
X(t) = g*1(t) A1(t) + g*2(t) A2(t) - ...
Hc(t) = -i sum_{i>j} (gi(t) g*j(t) Adi Aj - H.c.) / 2
"""

import qutip as qt
import numpy as np

def solve_cascaded_system(dims, Hs, gs, *args, **kwargs):
    """
    Args:
      dims (list): Dimensions of the individual systems.
      Hs (list): Hamiltonians of the individual systems
          as functions of time.
      gs (list): In/out couplings of the individual systems
          as functions of time.
      args (tuple): Additional arguments to be passed to qutip.mesolve.
      kwargs (dict): Additional arguments to be passed to qutip.mesolve.
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

    def c_ops(t, args):
        result = qt.tensor(*[qt.qzero(dim) for dim in dims])
        for i, gi in enumerate(gs):
            result += gi(t).conjugate()*A[i]
        return result

    kwargs["c_ops"] = c_ops
    return qt.mesolve(H, *args, **kwargs)
