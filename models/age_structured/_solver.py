'''Solver.'''

import functools

import numpy
import scipy.optimize
import scipy.sparse

from .. import _solver
from .. import _utility


# Common sparse array format.
_SPARSE_ARRAY = scipy.sparse.csr_array


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.a_step = self.t_step = model.a_step
        super().__init__(model)

    def _I(self):
        n = len(self.model.states)
        J = len(self.model.a)
        I = scipy.sparse.identity(n * J)
        return _SPARSE_ARRAY(I)

    def _zeros(self):
        J = len(self.model.a)
        zeros = {'11': _SPARSE_ARRAY((1, 1)),
                 '1J': _SPARSE_ARRAY((1, J)),
                 'J1': _SPARSE_ARRAY((J, 1)),
                 'JJ': _SPARSE_ARRAY((J, J))}
        return zeros

    def _beta(self):
        J = len(self.model.a)
        ones1J = numpy.ones((1, J))
        zeros1J = self.zeros['1J']
        beta = (
            self.model.transmission.rate
            * self.a_step
            * scipy.sparse.bmat([
                [zeros1J, zeros1J, zeros1J, ones1J, zeros1J]
            ])
        )
        return _SPARSE_ARRAY(beta)

    def _HqXX(self, q):
        return self._FqXW(q, 1)

    def _Hq(self, q):
        n = len(self.model.states)
        HqXX = self._HqXX(q)
        Hq = scipy.sparse.block_diag((HqXX, ) * n)
        return _SPARSE_ARRAY(Hq)

    def _FqXW(self, q, pi):
        J = len(self.model.a)
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(J)
        if q == 'new':
            diags = {0: pi}
        elif q == 'cur':
            diags = {-1: pi[:-1],
                     0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])}
        else:
            raise ValueError(f'{q=}!')
        return _utility.sparse.diags(diags)

    def _Fq(self, q):
        mu = self.model.death.rate(self.model.a)
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
        TqXW = self._TqXW(q)
        ZerosJJ = self.zeros['JJ']
        Tq = scipy.sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, - TqXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, TqXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return _SPARSE_ARRAY(Tq)

    def _BXW(self):
        J = len(self.model.a)
        nu = self.model.birth.maternity(self.model.a)
        BXW = scipy.sparse.lil_array((J, J))
        BXW[0] = nu
        return BXW

    def _B(self):
        BXW = self._BXW()
        ZerosJJ = self.zeros['JJ']
        B = scipy.sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, BXW],
            [BXW, BXW, BXW, BXW, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return _SPARSE_ARRAY(B)

    def _build_matrices(self):
        self.I = self._I()
        self.zeros = self._zeros()
        self.beta = self._beta()
        self.H_new = self._Hq('new')
        self.H_cur = self._Hq('cur')
        self.F_new = self._Fq('new')
        self.F_cur = self._Fq('cur')
        self.T_new = self._Tq('new')
        self.T_cur = self._Tq('cur')
        self.B = self._B()
        self.krylov_M = (self.H_new
                         + self.t_step / 2 * self.F_new)

    def _check_matrices(self):
        assert _utility.is_nonnegative(self.beta)
        assert _utility.is_Z_matrix(self.H_new)
        assert _utility.is_nonnegative(self.H_cur)
        assert _utility.is_Metzler_matrix(self.F_new)
        assert _utility.is_Metzler_matrix(self.T_new)
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
                    - self.t_step / 2 * self.beta @ y_new * self.T_new)
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
                                         + self.beta @ y_cur * self.T_cur
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
                                      + self.beta @ y_new * self.T_new
                                      + numpy.outer(self.T_new @ y_new,
                                                    self.beta)
                                      + b_mid * self.B))
        M_cur = (self.H_cur
                 + self.t_step / 2 * (self.F_cur
                                      + self.beta @ y_cur * self.T_cur
                                      + numpy.outer(self.T_cur @ y_cur,
                                                    self.beta)
                                      + b_mid * self.B))
        D = scipy.linalg.solve(M_new, M_cur,
                               overwrite_a=True,
                               overwrite_b=True)
        J = (D - self.I) / self.t_step
        return J


def solve(model, t_span, y_0,
          t=None, y=None, display=False):
    '''Solve the model.'''
    solver = Solver(model)
    return solver.solve(t_span, y_0,
                        t=t, y=y, display=display)
