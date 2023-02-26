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

    @functools.cached_property
    def Zeros(self):
        '''These are zero matrices of different sizes used in
        constructing the other matrices. `Zeros` is built on first use
        and then reused.'''
        J = len(self.model.a)
        Zeros = {
            '1J': sparse.array((1, J)),
            'JJ': sparse.array((J, J))
        }
        return Zeros

    def _MXW(self, q):
        '''Build HXX(q) and TXW(q).'''
        J = len(self.model.a)
        if q == 'new':
            diags = {
                0: numpy.ones(J)
            }
        elif q == 'cur':
            diags = {
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            }
        else:
            raise ValueError(f'{q=}!')
        MXW = sparse.diags_from_dict(diags)
        return MXW

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        J = len(self.model.a)
        size = n * J
        I = sparse.identity(size)
        return I

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        J = len(self.model.a)
        ones1J = numpy.ones((1, J))
        zeros1J = self.Zeros['1J']
        beta = (
            self.model.transmission.rate
            * self.a_step
            * sparse.hstack(
                [zeros1J, zeros1J, zeros1J, ones1J, zeros1J]
            )
        )
        return beta

    def _HXX(self, q):
        '''Build a diagonal block of H(q).'''
        HXX = self._MXW(q)
        return HXX

    def _H(self, q):
        '''Build the time step matrix H(q).'''
        n = len(self.model.states)
        HXX = self._HXX(q)
        H = sparse.block_diag([HXX] * n)
        return H

    def _FXW(self, q, pi):
        '''Build a block of F(q).'''
        J = len(self.model.a)
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(J)
        if q == 'new':
            diags = {
                0: pi
            }
        elif q == 'cur':
            diags = {
                -1: pi[:-1],
                0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])
            }
        else:
            raise ValueError(f'{q=}!')
        FXW = sparse.diags_from_dict(diags)
        return FXW

    def _F(self, q):
        '''Build the transition matrix F(q).'''
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
        '''Build a block of T(q).'''
        TXW = self._MXW(q)
        return TXW

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        TXW = self._TXW(q)
        ZerosJJ = self.Zeros['JJ']
        T = sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ,   - TXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ,     TXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return T

    def _BXW(self):
        '''Build a block of B.'''
        J = len(self.model.a)
        shape = (J, J)
        nu = self.model.birth.maternity(self.model.a)
        # The first row is `nu`.
        data = {
            (0, (None, )): nu
        }
        BXW = sparse.array_from_dict(data, shape=shape)
        return BXW

    def _B(self):
        '''Build the birth matrix B.'''
        BXW = self._BXW()
        ZerosJJ = self.Zeros['JJ']
        B = sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ,     BXW],
            [    BXW,     BXW,     BXW,     BXW, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return B
