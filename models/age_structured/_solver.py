'''Solver.'''

import functools

import numpy
import scipy.optimize
import scipy.sparse

from .. import _utility


# Common sparse array format.
_SPARSE_ARRAY = scipy.sparse.csr_array


class Solver:
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.model = model
        self.age_step = self.time_step = model.age_step
        self._build_matrices()

    def _beta(self):
        J = len(self.model.ages)
        zeros = _SPARSE_ARRAY((1, J))
        ones = numpy.ones((1, J))
        beta = (self.model.transmission.rate
                * scipy.sparse.bmat([[zeros, zeros, zeros, ones, zeros]]))
        return _SPARSE_ARRAY(beta)

    def _HqXX(self, q):
        return self._FqXW(q, 1)

    def _Hq(self, q):
        n = len(self.model.states)
        HqXX = self._HqXX(q)
        Hq = scipy.sparse.block_diag((HqXX, ) * n)
        return _SPARSE_ARRAY(Hq)

    def _FqXW(self, q, pi):
        J = len(self.model.ages)
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(J)
        if q == 0:
            diags = {0: pi}
        elif q == 1:
            diags = {-1: pi[:-1],
                     0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])}
        else:
            raise ValueError(f'{q=}!')
        return _utility.sparse.diags(diags)

    def _Fq(self, q):
        mu = self.model.death.rate(self.model.ages)
        omega = 1 / self.model.waning.mean
        rho = 1 / self.model.progression.mean
        gamma = 1 / self.model.recovery.mean
        FqXW = functools.partial(self._FqXW, q)
        Fq = scipy.sparse.bmat([
            [FqXW(- omega - mu), None, None, None, None],
            [FqXW(omega), FqXW(- mu), None, None, None],
            [None, None, FqXW(- rho - mu), None, None],
            [None, None, FqXW(rho), FqXW(- gamma - mu), None],
            [None, None, None, FqXW(gamma), FqXW(- mu)]
        ])
        return _SPARSE_ARRAY(Fq)

    def _TqXW(self, q):
        return self._FqXW(q, 1)

    def _Tq(self, q):
        J = len(self.model.ages)
        TqXW = self._TqXW(q)
        Zeros = _SPARSE_ARRAY((J, J))
        Tq = scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, - TqXW, Zeros, Zeros, Zeros],
            [Zeros, TqXW, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])
        return _SPARSE_ARRAY(Tq)

    def _BXW(self):
        J = len(self.model.ages)
        nu = self.model.birth.maternity(self.model.ages)
        BXW = scipy.sparse.lil_array((J, J))
        BXW[0] = nu
        return BXW

    def _B(self):
        J = len(self.model.ages)
        BXW = self._BXW()
        Zeros = _SPARSE_ARRAY((J, J))
        B = scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, BXW],
            [BXW, BXW, BXW, BXW, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])
        return _SPARSE_ARRAY(B)

    def _build_matrices(self):
        self.beta = self._beta()
        self.H0 = self._Hq(0)
        self.H1 = self._Hq(1)
        self.F0 = self._Fq(0)
        self.F1 = self._Fq(1)
        self.T0 = self._Tq(0)
        self.T1 = self._Tq(1)
        self.B = self._B()

    def _objective(self, y_new, HFB0, HFBTy1):
        lambdaT0 = (self.beta @ y_new) * self.T0
        HFBT0 = HFB0 - self.time_step / 2 * lambdaT0
        return HFBT0 @ y_new - HFBTy1

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        lambdaT1 = (self.beta @ y_cur) * self.T1
        t_mid = t_cur + 0.5 * self.time_step
        bB = self.model.birth.rate(t_mid) * self.B
        HFB0 = self.H0 - self.time_step / 2 * (self.F0 + bB)
        HFBT1 = self.H1 + self.time_step / 2 * (self.F1 + lambdaT1 + bB)
        HFBTy1 = HFBT1 @ y_cur
        result = scipy.optimize.root(self._objective, y_cur,
                                     args=(HFB0, HFBTy1))
        assert result.success, f't={t_cur}: {result}'
        y_new[:] = result.x

    def __call__(self, t_span, y_0,
                 t=None, y=None, _solution_wrap=True):
        '''Solve. `y` is storage for the solution, which will be built if not
        provided. `_solution_wrap=False` skips wrapping the solution in
        `model.Solution()` for speed.'''
        if t is None:
            t = _utility.build_t(*t_span, self.time_step)
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for ell in range(1, len(t)):
            self._step(t[ell - 1], y[ell - 1], y[ell])
        if _solution_wrap:
            return self.model.Solution(y, t)
        else:
            return y


def solve(model, t_span, y_0,
          t=None, y=None, _solution_wrap=True):
    '''Solve the `model` system of PDEs.'''
    solver = Solver(model)
    return solver(t_span, y_0,
                  t=t, y=y, _solution_wrap=_solution_wrap)
