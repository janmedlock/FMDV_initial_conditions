'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Solver(_model.solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = True

    _jacobian_method_default = 'dense'

    def __init__(self, model, **kwds):
        self.a_step = model.a_step
        self.a = model.a
        super().__init__(model, **kwds)

    @staticmethod
    def _get_a_step(t_step):
        a_step = t_step
        return a_step

    @functools.cached_property
    def Zeros(self):
        '''These are zero matrices of different sizes used in
        constructing the other matrices. `Zeros` is built on first use
        and then reused.'''
        J = len(self.a)
        Zeros = {
            '1J': _utility.sparse.array((1, J)),
            'JJ': _utility.sparse.array((J, J))
        }
        return Zeros

    @functools.cached_property
    def IXW(self):
        '''Build an identity matrix block. `IXW` is built on first use
        and then reused.'''
        J = len(self.a)
        IXW = _utility.sparse.identity(J)
        return IXW

    @functools.cached_property
    def LXW(self):
        '''Build the lag matrix. `LXW` is built on first use and then
        reused.'''
        J = len(self.a)
        diags = {
            -1: numpy.ones(J - 1),
            0: numpy.hstack([numpy.zeros(J - 1), 1])
        }
        LXW = _utility.sparse.diags_from_dict(diags)
        return LXW

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        J = len(self.a)
        I = _utility.sparse.identity(n * J)
        return I

    def _MXW(self, q):
        '''Build HXX(q) and TXW(q).'''
        if q == 'new':
            MXW = self.IXW
        elif q == 'cur':
            MXW = self.LXW
        else:
            raise ValueError(f'{q=}!')
        return MXW

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        J = len(self.a)
        ones1J = numpy.ones((1, J))
        zeros1J = self.Zeros['1J']
        beta = (
            self.model.parameters.transmission.rate
            * self.a_step
            * _utility.sparse.hstack(
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
        H = _utility.sparse.block_diag([HXX] * n)
        return H

    def _FXW(self, q, pi):
        '''Build a block of F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(pi):
            J = len(self.a)
            pi *= numpy.ones(J)
        FXW = _utility.sparse.diags(pi)
        if q == 'cur':
            FXW = self.LXW @ FXW
        return FXW

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        mu = self.model.parameters.death.rate(self.a)
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        FXW = functools.partial(self._FXW, q)
        F = _utility.sparse.bmat([
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
        T = _utility.sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ,   - TXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ,     TXW, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return T

    def _BXW(self):
        '''Build a block of B.'''
        J = len(self.a)
        nu = self.model.parameters.birth.maternity(self.a)
        tau = _utility.sparse.array(self.a_step * nu)
        b = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.a_step},
            shape=(J, 1)
        )
        BXW = b @ tau
        return BXW

    def _B(self):
        '''Build the birth matrix B.'''
        BXW = self._BXW()
        ZerosJJ = self.Zeros['JJ']
        B = _utility.sparse.bmat([
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ,     BXW],
            [    BXW,     BXW,     BXW,     BXW, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ],
            [ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ, ZerosJJ]
        ])
        return B
