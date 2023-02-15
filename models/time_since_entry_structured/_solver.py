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
        self.z_step = self.time_step = model.z_step
        self._build_matrices()

    def _beta(self):
        K = len(self.model.z)
        zeros = _SPARSE_ARRAY((1, K))
        ones = numpy.ones((1, K))
        beta = (self.model.transmission.rate
                * self.z_step
                * scipy.sparse.bmat([[zeros, 0, zeros, ones, 0]]))
        return _SPARSE_ARRAY(beta)

    def _Hqyy(self, q):
        return self._Fqyy(q, 1)

    def _Hq(self, q):
        Hqyy = self._Hqyy(q)
        Hq = scipy.sparse.block_diag((Hqyy, [[1]], Hqyy, Hqyy, [[1]]))
        return _SPARSE_ARRAY(Hq)

    def _Fqyy(self, q, psi):
        K = len(self.model.z)
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        if q == 0:
            diags = {0: psi}
        elif q == 1:
            diags = {-1: psi[:-1],
                     0: numpy.hstack([numpy.zeros(K - 1), psi[-1]])}
        else:
            raise ValueError(f'{q=}!')
        return _utility.sparse.diags(diags)

    def _Fyz(self, psi):
        K = len(self.model.z)
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        Fyz = scipy.sparse.lil_array((K, K))
        Fyz[0] = psi
        return Fyz

    def _fXy(self, psi):
        if numpy.isscalar(psi):
            K = len(self.model.z)
            psi = psi * numpy.ones(K)
        return self.z_step * psi

    def _Fq(self, q):
        mu = self.model.death_rate_mean
        omega = self.model.waning.rate(self.model.z)
        rho = self.model.progression.rate(self.model.z)
        gamma = self.model.recovery.rate(self.model.z)
        Fqyy = functools.partial(self._Fqyy, q)
        Fyz = self._Fyz
        fXy = self._fXy
        Fq = scipy.sparse.bmat([
            [Fqyy(- (omega + mu)), None, None, None, None],
            [fXy(omega), [[- mu]], None, None, None],
            [None, None, Fqyy(- (rho + mu)), None, None],
            [None, None, Fyz(rho), Fqyy(- (gamma + mu)), None],
            [None, None, None, fXy(gamma), [[- mu]]]
        ])
        return _SPARSE_ARRAY(Fq)

    def _tyX(self):
        K = len(self.model.z)
        tyX = scipy.sparse.lil_array((K, 1))
        tyX[0] = 1 / self.z_step
        return tyX

    def _T(self):
        K = len(self.model.z)
        tyX = self._tyX()
        Zeros = _SPARSE_ARRAY((K, K))
        zerosK1 = _SPARSE_ARRAY((K, 1))
        zeros1K = _SPARSE_ARRAY((1, K))
        T = scipy.sparse.bmat([
            [Zeros, zerosK1, Zeros, Zeros, zerosK1],
            [zeros1K, [[- 1]], zeros1K, zeros1K, [[0]]],
            [Zeros, tyX, Zeros, Zeros, zerosK1],
            [Zeros, zerosK1, Zeros, Zeros, zerosK1],
            [zeros1K, [[0]], zeros1K, zeros1K, [[0]]]
        ])
        return _SPARSE_ARRAY(T)

    def _byX(self):
        K = len(self.model.z)
        byX = scipy.sparse.lil_array((K, 1))
        byX[0] = 1 / self.z_step
        return byX

    def _bXy(self):
        K = len(self.model.z)
        return 1 / self.z_step * numpy.ones((1, K))

    def _B(self):
        K = len(self.model.z)
        byX = self._byX()
        bXy = self._bXy()
        Zeros = _SPARSE_ARRAY((K, K))
        zerosK1 = _SPARSE_ARRAY((K, 1))
        zeros1K = _SPARSE_ARRAY((1, K))
        B = scipy.sparse.bmat([
            [Zeros, zerosK1, Zeros, Zeros, byX],
            [bXy, [[1]], bXy, bXy, [[0]]],
            [Zeros, zerosK1, Zeros, Zeros, zerosK1],
            [Zeros, zerosK1, Zeros, Zeros, zerosK1],
            [zeros1K, [[0]], zeros1K, zeros1K, [[0]]]
        ])
        return _SPARSE_ARRAY(B)

    def _build_matrices(self):
        self.beta = self._beta()
        self.H0 = self._Hq(0)
        self.H1 = self._Hq(1)
        self.F0 = self._Fq(0)
        self.F1 = self._Fq(1)
        self.T = self._T()
        self.B = self._B()

    def _objective(self, y_new, HFB0, HFBTy1):
        lambdaT0 = (self.beta @ y_new) * self.T
        HFBT0 = HFB0 - self.time_step / 2 * lambdaT0
        return HFBT0 @ y_new - HFBTy1

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        lambdaT1 = (self.beta @ y_cur) * self.T
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
