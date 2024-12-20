'''Solver.'''

import functools

import numpy

from .. import _model


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    sparse = False

    _jacobian_method_default = 'base'

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        blocks = [0] * len(self.model.states)
        infectious = self.model.states.index('infectious')
        blocks[infectious] = 1
        return (
            self.model.parameters.transmission.rate
            * numpy.array(blocks).reshape((1, -1))
        )

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return numpy.identity(len(self.model.states))

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        return self.I

    @functools.cached_property
    def _F(self):  # pylint: disable=invalid-name
        '''F is independent of q, so build it once and reuse it.'''
        mu = self.model.parameters.death_rate_mean
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        return numpy.array([
            [- omega - mu, 0, 0, 0, 0],
            [omega, - mu, 0, 0, 0],
            [0, 0, - rho - mu, 0, 0],
            [0, 0, rho, - gamma - mu, 0],
            [0, 0, 0, gamma, - mu]
        ])

    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''
        return self._F

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        return numpy.array([
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

    @functools.cached_property
    def _T_(self):  # pylint: disable=invalid-name
        '''T is independent of q, so build it once and reuse.'''
        return numpy.array([
            [0, 0,   0, 0, 0],
            [0, - 1, 0, 0, 0],
            [0, 1,   0, 0, 0],
            [0, 0,   0, 0, 0],
            [0, 0,   0, 0, 0]
        ])

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        return self._T_
