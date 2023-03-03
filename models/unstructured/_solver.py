'''Solver.'''

import functools

import numpy

from .. import _model


class Solver(_model.solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = False

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        I = numpy.identity(n)
        return I

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        beta = (
            self.model.transmission.rate
            * numpy.array([
                [0, 0, 0, 1, 0]
            ])
        )
        return beta

    def _H(self, q):
        '''Build the time step matrix H(q).'''
        H = self.I
        return H

    @functools.cached_property
    def _F_(self):
        '''F is independent of q. `_F_` is built on first use and then
        reused.'''
        mu = self.model.death_rate_mean
        omega = 1 / self.model.waning.mean
        rho = 1 / self.model.progression.mean
        gamma = 1 / self.model.recovery.mean
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

    '''T is independent of q. Build _T_ once and reuse.'''
    _T_ = numpy.array([
        [0, 0, 0, 0, 0],
        [0, - 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        T = self._T_
        return T

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
