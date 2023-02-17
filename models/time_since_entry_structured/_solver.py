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
        self._check_matrices()

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
        HXX = [[1]]
        Hq = scipy.sparse.block_diag((Hqyy, HXX, Hqyy, Hqyy, HXX))
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

    @staticmethod
    def _fXX(pi):
        return [[pi]]

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
        fXX = self._fXX
        Fyz = self._Fyz
        fXy = self._fXy
        Fq = scipy.sparse.bmat([
            [Fqyy(- omega - mu), None, None, None, None],
            [fXy(omega), fXX(- mu), None, None, None],
            [None, None, Fqyy(- rho - mu), None, None],
            [None, None, Fyz(rho), Fqyy(- gamma - mu), None],
            [None, None, None, fXy(gamma), fXX(- mu)]
        ])
        return _SPARSE_ARRAY(Fq)

    def _tyX(self):
        K = len(self.model.z)
        tyX = scipy.sparse.lil_array((K, 1))
        tyX[0] = 1 / self.z_step
        return tyX

    def _T(self):
        K = len(self.model.z)
        tXX = numpy.array([[1]])
        tyX = self._tyX()
        zerosKK = _SPARSE_ARRAY((K, K))
        zerosK1 = _SPARSE_ARRAY((K, 1))
        zeros1K = _SPARSE_ARRAY((1, K))
        zeros11 = _SPARSE_ARRAY((1, 1))
        T = scipy.sparse.bmat([
            [zerosKK, zerosK1, zerosKK, zerosKK, zerosK1],
            [zeros1K, - tXX, zeros1K, zeros1K, zeros11],
            [zerosKK, tyX, zerosKK, zerosKK, zerosK1],
            [zerosKK, zerosK1, zerosKK, zerosKK, zerosK1],
            [zeros1K, zeros11, zeros1K, zeros1K, zeros11]
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
        bXX = [[1]]
        byX = self._byX()
        bXy = self._bXy()
        zerosKK = _SPARSE_ARRAY((K, K))
        zerosK1 = _SPARSE_ARRAY((K, 1))
        zeros1K = _SPARSE_ARRAY((1, K))
        zeros11 = _SPARSE_ARRAY((1, 1))
        B = scipy.sparse.bmat([
            [zerosKK, zerosK1, zerosKK, zerosKK, byX],
            [bXy, bXX, bXy, bXy, zeros11],
            [zerosKK, zerosK1, zerosKK, zerosKK, zerosK1],
            [zerosKK, zerosK1, zerosKK, zerosKK, zerosK1],
            [zeros1K, zeros11, zeros1K, zeros1K, zeros11]
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
        self.krylov_M = self.H0 + self.time_step / 2 * self.F0

    def _check_matrices(self):
        assert _utility.is_nonnegative(self.beta)
        assert _utility.is_Z_matrix(self.H0)
        assert _utility.is_nonnegative(self.H1)
        assert _utility.is_Metzler_matrix(self.F0)
        assert _utility.is_Metzler_matrix(self.T)
        assert _utility.is_Metzler_matrix(self.B)
        assert _utility.is_nonnegative(self.B)
        HFB0 = (self.H0
                - self.time_step / 2 * (self.F0
                                        + self.model.birth.rate_max * self.B))
        assert _utility.is_M_matrix(HFB0)
        HFB1 = (self.H1
                + self.time_step / 2 * (self.F1
                                        + self.model.birth.rate_min * self.B))
        assert _utility.is_nonnegative(HFB1)

    def _objective(self, y_new, HFB0, HFBTy1):
        lambdaT0 = (self.beta @ y_new) * self.T
        HFBT0 = HFB0 - self.time_step / 2 * lambdaT0
        return HFBT0 @ y_new - HFBTy1

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        t_mid = t_cur + 0.5 * self.time_step
        bB = self.model.birth.rate(t_mid) * self.B
        HFB0 = self.H0 - self.time_step / 2 * (self.F0 + bB)
        lambdaT1 = (self.beta @ y_cur) * self.T
        HFBT1 = self.H1 + self.time_step / 2 * (self.F1 + lambdaT1 + bB)
        HFBTy1 = HFBT1 @ y_cur
        result = scipy.optimize.root(
            self._objective, y_cur, args=(HFB0, HFBTy1),
            method='krylov',
            options=dict(jac_options=dict(inner_M=self.krylov_M))
        )
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
