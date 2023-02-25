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
            * sparse.hstack(
                [zeros1J, zeros1J, zeros1J, ones1J, zeros1J]
            )
        )
        return beta

    def _HXX(self, q):
        HXX = self._FXW(q, 1)
        return HXX

    def _H(self, q):
        n = len(self.model.states)
        HXX = self._HXX(q)
        diag = (HXX, ) * n
        H = sparse.block_diag(diag)
        return H

    def _FXW(self, q, pi):
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
        FXW = sparse.diags_from_dict(diags)
        return FXW

    def _F(self, q):
        mu = self.model.death.rate(self.model.a)
        omega = 1 / self.model.waning.mean
        rho = 1 / self.model.progression.mean
        gamma = 1 / self.model.recovery.mean
        FXW = functools.partial(self._FXW, q)
        F = sparse.bmat([
            [FXW(- omega - mu), None, None, None, None],
            [FXW(omega), FXW(- mu), None, None, None],
            [None, None, FXW(- rho - mu), None, None],
            [None, None, FXW(rho), FXW(- gamma - mu), None],
            [None, None, None, FXW(gamma), FXW(- mu)]
        ])
        return F

    def _TXW(self, q):
        TXW = self._FXW(q, 1)
        return TXW

    def _T(self, q):
        TXW = self._TXW(q)
        ZerosJJ = self.zeros['JJ']
        T = sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, - TXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, TXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return T

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
