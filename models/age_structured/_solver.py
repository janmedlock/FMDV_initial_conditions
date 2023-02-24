'''Solver.'''

import functools

import numpy

from .. import _solver
from .._utility import sparse


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = True

    def __init__(self, model):
        self.a_step = self.t_step = model.a_step
        super().__init__(model)
        self.krylov_M = (self.H_new
                         + self.t_step / 2 * self.F_new)

    def _I(self):
        n = len(self.model.states)
        J = len(self.model.a)
        size = n * J
        I = sparse.identity(size)
        return I

    # Build `zeros` on first use and then reuse.
    @functools.cached_property
    def zeros(self):
        J = len(self.model.a)
        zeros = {'11': sparse.array((1, 1)),
                 '1J': sparse.array((1, J)),
                 'J1': sparse.array((J, 1)),
                 'JJ': sparse.array((J, J))}
        return zeros

    def _beta(self):
        J = len(self.model.a)
        ones1J = numpy.ones((1, J))
        zeros1J = self.zeros['1J']
        beta = (
            self.model.transmission.rate
            * self.a_step
            * sparse.bmat([
                [zeros1J, zeros1J, zeros1J, ones1J, zeros1J]
            ])
        )
        return beta

    def _HqXX(self, q):
        HqXX = self._FqXW(q, 1)
        return HqXX

    def _Hq(self, q):
        n = len(self.model.states)
        HqXX = self._HqXX(q)
        diag = (HqXX, ) * n
        Hq = sparse.block_diag(diag)
        return Hq

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
        FqXW = sparse.diags_from_dict(diags)
        return FqXW

    def _Fq(self, q):
        mu = self.model.death.rate(self.model.a)
        omega = 1 / self.model.waning.mean
        rho = 1 / self.model.progression.mean
        gamma = 1 / self.model.recovery.mean
        FqXW = functools.partial(self._FqXW, q)
        Fq = sparse.bmat([
            [FqXW(- omega - mu), None, None, None, None],
            [FqXW(omega), FqXW(- mu), None, None, None],
            [None, None, FqXW(- rho - mu), None, None],
            [None, None, FqXW(rho), FqXW(- gamma - mu), None],
            [None, None, None, FqXW(gamma), FqXW(- mu)]
        ])
        return Fq

    def _TqXW(self, q):
        TqXW = self._FqXW(q, 1)
        return TqXW

    def _Tq(self, q):
        TqXW = self._TqXW(q)
        ZerosJJ = self.zeros['JJ']
        Tq = sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, - TqXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, TqXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return Tq

    def _BXW(self):
        J = len(self.model.a)
        shape = (J, J)
        nu = self.model.birth.maternity(self.model.a)
        # The first row is `nu`.
        data = {(0, (None, )): nu}
        BXW = sparse.array_from_dict(data, shape=shape)
        return BXW

    def _B(self):
        BXW = self._BXW()
        ZerosJJ = self.zeros['JJ']
        B = sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, BXW],
            [BXW, BXW, BXW, BXW, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return B
