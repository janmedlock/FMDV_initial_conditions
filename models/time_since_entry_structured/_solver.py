'''Solver.'''

import functools

import numpy
import scipy.optimize
import scipy.sparse.linalg

from .. import _solver
from .. import _utility


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = True

    def __init__(self, model):
        self.z_step = self.t_step = model.z_step
        super().__init__(model)

    def _I(self):
        n = len(self.model.states)
        m = len(self.model.states_with_z)
        K = len(self.model.z)
        size = m * K + (n - m)
        I = _utility.sparse.identity(size)
        return I

    def _zeros(self):
        K = len(self.model.z)
        zeros = {'11': _utility.sparse.array((1, 1)),
                 '1K': _utility.sparse.array((1, K)),
                 'K1': _utility.sparse.array((K, 1)),
                 'KK': _utility.sparse.array((K, K))}
        return zeros

    def _beta(self):
        K = len(self.model.z)
        ones1K = numpy.ones((1, K))
        zeros = self.zeros
        beta = (
            self.model.transmission.rate
            * self.z_step
            * _utility.sparse.bmat([
                [zeros['1K'], zeros['11'], zeros['1K'], ones1K, zeros['11']]
            ])
        )
        return beta

    def _Hqyy(self, q):
        Hqyy = self._Fqyy(q, 1)
        return Hqyy

    def _Hq(self, q):
        Hqyy = self._Hqyy(q)
        HXX = [[1]]
        Hq = _utility.sparse.block_diag((Hqyy, HXX, Hqyy, Hqyy, HXX))
        return Hq

    def _Fqyy(self, q, psi):
        K = len(self.model.z)
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        if q == 'new':
            diags = {0: psi}
        elif q == 'cur':
            diags = {-1: psi[:-1],
                     0: numpy.hstack([numpy.zeros(K - 1), psi[-1]])}
        else:
            raise ValueError(f'{q=}!')
        Fqyy = _utility.sparse.diags_from_dict(diags)
        return Fqyy

    @staticmethod
    def _fXX(pi):
        fXX = [[pi]]
        return fXX

    def _Fyz(self, psi):
        K = len(self.model.z)
        shape = (K, K)
        # The first row is `psi`.
        data = {(0, (None, )): psi}
        Fyz = _utility.sparse.array_from_dict(data, shape=shape)
        return Fyz

    def _fXy(self, psi):
        if numpy.isscalar(psi):
            K = len(self.model.z)
            psi = psi * numpy.ones(K)
        fXy = self.z_step * psi
        return fXy

    def _get_rate(self, which):
        param = getattr(self.model, which)
        rate = param.rate(self.model.z)
        return _utility.rate_make_finite(rate)

    def _Fq(self, q):
        mu = self.model.death_rate_mean
        omega = self._get_rate('waning')
        rho = self._get_rate('progression')
        gamma = self._get_rate('recovery')
        Fqyy = functools.partial(self._Fqyy, q)
        fXX = self._fXX
        Fyz = self._Fyz
        fXy = self._fXy
        Fq = _utility.sparse.bmat([
            [Fqyy(- omega - mu), None, None, None, None],
            [fXy(omega), fXX(- mu), None, None, None],
            [None, None, Fqyy(- rho - mu), None, None],
            [None, None, Fyz(rho), Fqyy(- gamma - mu), None],
            [None, None, None, fXy(gamma), fXX(- mu)]
        ])
        return Fq

    def _tyX(self):
        K = len(self.model.z)
        shape = (K, 1)
        # The first entry is `1 / self.z_step`.
        data = {(0, 0): 1 / self.z_step}
        tyX = _utility.sparse.array_from_dict(data, shape=shape)
        return tyX

    def _T(self):
        tXX = numpy.array([[1]])
        tyX = self._tyX()
        zeros = self.zeros
        T = _utility.sparse.bmat([
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['1K'], - tXX, zeros['1K'], zeros['1K'], zeros['11']],
            [zeros['KK'], tyX, zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['1K'], zeros['11'], zeros['1K'], zeros['1K'], zeros['11']]
        ])
        return T

    def _byX(self):
        K = len(self.model.z)
        shape = (K, 1)
        # The first entry is `1 / self.z_step`.
        data = {(0, 0): 1 / self.z_step}
        byX = _utility.sparse.array_from_dict(data, shape=shape)
        return byX

    def _bXy(self):
        K = len(self.model.z)
        shape = (1, K)
        bXy = self.z_step * numpy.ones(shape)
        return bXy

    def _B(self):
        bXX = [[1]]
        byX = self._byX()
        bXy = self._bXy()
        zeros = self.zeros
        B = _utility.sparse.bmat([
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], byX],
            [bXy, bXX, bXy, bXy, zeros['11']],
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['1K'], zeros['11'], zeros['1K'], zeros['1K'], zeros['11']]
        ])
        return B

    def _build_matrices(self):
        self.I = self._I()
        self.zeros = self._zeros()
        self.beta = self._beta()
        self.H_new = self._Hq('new')
        self.H_cur = self._Hq('cur')
        self.F_new = self._Fq('new')
        self.F_cur = self._Fq('cur')
        self.T = self._T()
        self.B = self._B()
        self.krylov_M = (self.H_new
                         + self.t_step / 2 * self.F_new)

    def _check_matrices(self):
        assert _utility.is_nonnegative(self.beta)
        assert _utility.is_Z_matrix(self.H_new)
        assert _utility.is_nonnegative(self.H_cur)
        assert _utility.is_Metzler_matrix(self.F_new)
        assert _utility.is_Metzler_matrix(self.T)
        assert _utility.is_Metzler_matrix(self.B)
        assert _utility.is_nonnegative(self.B)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + self.model.birth.rate_max * self.B))
        assert _utility.is_M_matrix(HFB_new)
        HFB_cur = (self.H_cur
                   + self.t_step / 2 * (self.F_cur
                                        + self.model.birth.rate_min * self.B))
        assert _utility.is_nonnegative(HFB_cur)

    def _objective(self, y_new, HFB_new, HFTBy_cur):
        '''Helper for `.step()`.'''
        HFTB_new = (HFB_new
                    - self.t_step / 2 * self.beta @ y_new * self.T)
        return HFTB_new @ y_new - HFTBy_cur

    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + b_mid * self.B))
        HFTB_cur = (self.H_cur
                    + self.t_step / 2 * (self.F_cur
                                         + self.beta @ y_cur * self.T
                                         + b_mid * self.B))
        HFTBy_cur = HFTB_cur @ y_cur
        y_new_guess = y_cur
        result = scipy.optimize.root(
            self._objective, y_new_guess,
            args=(HFB_new, HFTBy_cur),
            method='krylov',
            options=dict(jac_options=dict(inner_M=self.krylov_M))
        )
        assert result.success, f'{t_cur=}\n{result=}'
        y_new = result.x
        return y_new

    def jacobian(self, t_cur, y_cur, y_new):
        '''The Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        # Compute `D`, the derivative of `y_cur` with respect to `y_new`,
        # which is `M_new @ D = M_cur`.
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        M_new = (self.H_new
                 - self.t_step / 2 * (self.F_new
                                      + self.beta @ y_new * self.T
                                      + numpy.outer(self.T @ y_new, self.beta)
                                      + b_mid * self.B))
        M_cur = (self.H_cur
                 + self.t_step / 2 * (self.F_cur
                                      + self.beta @ y_cur * self.T
                                      + numpy.outer(self.T @ y_cur, self.beta)
                                      + b_mid * self.B))
        D = scipy.sparse.linalg.spsolve(M_new, M_cur)
        J = (D - self.I) / self.t_step
        return J


def solve(model, t_span, y_0,
          t=None, y=None, display=False):
    '''Solve the model.'''
    solver = Solver(model)
    return solver.solve(t_span, y_0,
                        t=t, y=y, display=display)
