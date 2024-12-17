'''Solver.'''

import functools

import numpy

from .. import _model


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _sparse = False

    _jacobian_method_default = 'base'

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        n = len(self.model.states)
        blocks = [0] * n
        infectious = self.model.states.index('infectious')
        blocks[infectious] = 1
        beta = (self.model.parameters.transmission.rate
                * numpy.array(blocks).reshape((1, n)))
        return beta

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        I = numpy.identity(n)
        return I

    def _H(self, q):
        '''Build the time-step matrix H(q).'''
        H = self.I
        return H

    @functools.cached_property
    def _F_(self):
        '''F is independent of q, so build it once and reuse it.'''
        mu = self.model.parameters.death_rate_mean
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        F = numpy.array([
            [- omega - mu, 0, 0, 0, 0],
            [omega, - mu, 0, 0, 0],
            [0, 0, - rho - mu, 0, 0],
            [0, 0, rho, - gamma - mu, 0],
            [0, 0, 0, gamma, - mu]
        ])
        return F

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        F = self._F_
        return F

    @staticmethod
    def _B():
        '''Build the birth matrix B.'''
        B = numpy.array([
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return B

    @functools.cached_property
    def _T_(self):
        '''T is independent of q, so build it once and reuse.'''
        T = numpy.array([
            [0, 0, 0, 0, 0],
            [0, - 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return T

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        T = self._T_
        return T
