'''Solver.'''

import functools

import numpy

from . import _population
from .. import _model, _utility


class Solver(_population.solver.Base, _model.solver.Solver):
    '''Crank–Nicolson solver.'''

    sparse = True

    _jacobian_method_default = 'dense'

    def __init__(self, model, **kwds):
        super().__init__(model, model.parameters.age_max, **kwds)

    @property
    def _iota(self):
        '''A block for integrating over age.'''
        return self._integration_vector(len(self.a), self.a_step)

    @functools.cached_property
    def Zeros(self):  # pylint: disable=invalid-name
        '''Zero matrix used in constructing the other matrices.'''
        len_X = len(self.a)  # pylint: disable=invalid-name
        return _utility.sparse.Array((len_X, len_X))

    @property
    def _I_XW(self):  # pylint: disable=invalid-name
        '''The identity matrix block.'''
        return self._I_a

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''Build the identity matrix.'''
        return _utility.sparse.block_diag(
            [self._I_XW] * len(self.model.states)
        )

    @property
    def _L_XW(self):  # pylint: disable=invalid-name
        '''The lag matrix block.'''
        return self._L_a

    def _H_XX(self, q):  # pylint: disable=invalid-name
        '''The diagonal block of H(q).'''
        return self._H_a(q)

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        return _utility.sparse.block_diag(
            [self._H_XX(q)] * len(self.model.states)
        )

    def _F_XW(self, q, pi):  # pylint: disable=invalid-name
        '''A block of F(q).'''
        if q not in self._q_vals:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(pi):
            pi *= numpy.ones(len(self.a))
        F_XW = _utility.sparse.diags(pi)  # pylint: disable=invalid-name
        if q == 'cur':
            F_XW = self._L_XW @ F_XW  # pylint: disable=invalid-name
        return F_XW

    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''
        mu = self.model.parameters.death.rate(self.a)
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        # pylint: disable-next=invalid-name
        F_XW = functools.partial(self._F_XW, q)
        return _utility.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_XW(- rho - mu), None, None],
            [None, None, F_XW(rho), F_XW(- gamma - mu), None],
            [None, None, None, F_XW(gamma), F_XW(- mu)]
        ])

    @property
    def _B_XW(self):  # pylint: disable=invalid-name
        '''The block of B.'''
        return self._B_a

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        B_XW = self._B_XW  # pylint: disable=invalid-name
        Zeros = self.Zeros  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [None,  None, None, None, B_XW],
            [B_XW,  B_XW, B_XW, B_XW, None],
            [Zeros, None, None, None, None],
            [Zeros, None, None, None, None],
            [Zeros, None, None, None, None]
        ])

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        zeros = _utility.sparse.Array((1, len(self.a)))
        blocks = [zeros] * len(self.model.states)
        infectious = self.model.states.index('infectious')
        blocks[infectious] = self._iota
        return (
            self.model.parameters.transmission.rate
            * _utility.sparse.hstack(blocks)
        )

    def _T_XW(self, q):  # pylint: disable=invalid-name
        '''A block of T(q).'''
        return self._H_XX(q)

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        T_XW = self._T_XW(q)  # pylint: disable=invalid-name
        Zeros = self.Zeros  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [Zeros, None,   Zeros, None,  None],
            [None,  - T_XW, None,  None,  None],
            [None,  T_XW,   None,  None,  None],
            [None,  None,   None,  Zeros, None],
            [None,  None,   None,  None,  Zeros]
        ])
