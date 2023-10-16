'''Solver.'''

import functools

import numpy

from . import _base
from .. import _model, _utility


class Solver(_base.Solver, _model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _sparse = True

    _jacobian_method_default = 'dense'

    def __init__(self, model, **kwds):
        self.a = model.a
        super().__init__(model, **kwds)

    @functools.cached_property
    def Zeros(self):
        '''Zero matrix used in constructing the other matrices.'''
        J = len(self.a)
        Zeros = _utility.sparse.Array((J, J))
        return Zeros

    @functools.cached_property
    def I_XW(self):
        '''Build an identity matrix block.'''
        J = len(self.a)
        I_XW = _utility.sparse.identity(J)
        return I_XW

    @functools.cached_property
    def L_XW(self):
        '''Build the lag matrix.'''
        J = len(self.a)
        diags = {
            -1: numpy.ones(J - 1),
            0: numpy.hstack([numpy.zeros(J - 1), 1])
        }
        L_XW = _utility.sparse.diags_from_dict(diags)
        return L_XW

    def _iota(self):
        '''Build a block for integrating over age.'''
        J = len(self.a)
        iota = self.a_step * numpy.ones((1, J))
        return iota

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        I = _utility.sparse.block_diag([self.I_XW] * n)
        return I

    def _M_XW(self, q):
        '''Build H_XX(q) and T_XW(q).'''
        if q == 'new':
            M_XW = self.I_XW
        elif q == 'cur':
            M_XW = self.L_XW
        else:
            raise ValueError(f'{q=}!')
        return M_XW

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        n = len(self.model.states)
        J = len(self.a)
        zeros = _utility.sparse.Array((1, J))
        blocks = [zeros] * n
        infectious = self.model.states.index('infectious')
        blocks[infectious] = self._iota()
        beta = (self.model.parameters.transmission.rate
                * _utility.sparse.hstack(blocks))
        return beta

    def _H_XX(self, q):
        '''Build a diagonal block of H(q).'''
        H_XX = self._M_XW(q)
        return H_XX

    def _H(self, q):
        '''Build the time-step matrix H(q).'''
        n = len(self.model.states)
        H_XX = self._H_XX(q)
        H = _utility.sparse.block_diag([H_XX] * n)
        return H

    def _F_XW(self, q, pi):
        '''Build a block of F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(pi):
            J = len(self.a)
            pi *= numpy.ones(J)
        F_XW = _utility.sparse.diags(pi)
        if q == 'cur':
            F_XW = self.L_XW @ F_XW
        return F_XW

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        mu = self.model.parameters.death.rate(self.a)
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        F_XW = functools.partial(self._F_XW, q)
        F = _utility.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_XW(- rho - mu), None, None],
            [None, None, F_XW(rho), F_XW(- gamma - mu), None],
            [None, None, None, F_XW(gamma), F_XW(- mu)]
        ])
        return F

    def _T_XW(self, q):
        '''Build a block of T(q).'''
        T_XW = self._M_XW(q)
        return T_XW

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        T_XW = self._T_XW(q)
        Zeros = self.Zeros
        T = _utility.sparse.bmat([
            [Zeros,   None, Zeros,  None,  None],
            [ None, - T_XW,  None,  None,  None],
            [ None,   T_XW,  None,  None,  None],
            [ None,   None,  None, Zeros,  None],
            [ None,   None,  None,  None, Zeros]
        ])
        return T

    def _B_XW(self):
        '''Build a block of B.'''
        J = len(self.a)
        nu = self.model.parameters.birth.maternity(self.a)
        tau = _utility.sparse.Array(self.a_step * nu)
        b = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.a_step},
            shape=(J, 1)
        )
        B_XW = b @ tau
        return B_XW

    def _B(self):
        '''Build the birth matrix B.'''
        B_XW = self._B_XW()
        Zeros = self.Zeros
        B = _utility.sparse.bmat([
            [ None, None, None, None, B_XW],
            [ B_XW, B_XW, B_XW, B_XW, None],
            [Zeros, None, None, None, None],
            [Zeros, None, None, None, None],
            [Zeros, None, None, None, None]
        ])
        return B
