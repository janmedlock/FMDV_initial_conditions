'''Solver.'''

import functools

import numpy

from .. import _solver
from .. import _utility
from .._utility import sparse


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = True

    def __init__(self, model):
        self.z_step = self.t_step = model.z_step
        super().__init__(model)
        self.krylov_M = (self.H_new
                         + self.t_step / 2 * self.F_new)

    def _I(self):
        n = len(self.model.states)
        m = len(self.model.states_with_z)
        K = len(self.model.z)
        size = m * K + (n - m)
        I = sparse.identity(size)
        return I

    # Build `zeros` on first use and then reuse.
    @functools.cached_property
    def zeros(self):
        K = len(self.model.z)
        zeros = {'11': sparse.array((1, 1)),
                 '1K': sparse.array((1, K)),
                 'K1': sparse.array((K, 1)),
                 'KK': sparse.array((K, K))}
        return zeros

    def _beta(self):
        K = len(self.model.z)
        ones1K = numpy.ones((1, K))
        zeros = self.zeros
        beta = (
            self.model.transmission.rate
            * self.z_step
            * sparse.bmat([
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
        Hq = sparse.block_diag((Hqyy, HXX, Hqyy, Hqyy, HXX))
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
        Fqyy = sparse.diags_from_dict(diags)
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
        Fyz = sparse.array_from_dict(data, shape=shape)
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
        Fq = sparse.bmat([
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
        tyX = sparse.array_from_dict(data, shape=shape)
        return tyX

    # Build `_T` once and then reuse.
    @functools.cached_property
    def _T(self):
        tXX = numpy.array([[1]])
        tyX = self._tyX()
        zeros = self.zeros
        T = sparse.bmat([
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['1K'], - tXX, zeros['1K'], zeros['1K'], zeros['11']],
            [zeros['KK'], tyX, zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['1K'], zeros['11'], zeros['1K'], zeros['1K'], zeros['11']]
        ])
        return T

    def _Tq(self, q):
        # `Tq` is independent of `q`.
        Tq = self._T
        return Tq

    def _byX(self):
        K = len(self.model.z)
        shape = (K, 1)
        # The first entry is `1 / self.z_step`.
        data = {(0, 0): 1 / self.z_step}
        byX = sparse.array_from_dict(data, shape=shape)
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
        B = sparse.bmat([
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], byX],
            [bXy, bXX, bXy, bXy, zeros['11']],
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['KK'], zeros['K1'], zeros['KK'], zeros['KK'], zeros['K1']],
            [zeros['1K'], zeros['11'], zeros['1K'], zeros['1K'], zeros['11']]
        ])
        return B
