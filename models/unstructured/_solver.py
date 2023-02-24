'''Solver.'''

import functools

import numpy

from .. import _solver


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = False

    def __init__(self, model):
        self.t_step = model.t_step
        super().__init__(model)

    def _I(self):
        n = len(self.model.states)
        I = numpy.identity(n)
        return I

    def _beta(self):
        beta = (
            self.model.transmission.rate
            * numpy.array([
                [0, 0, 0, 1, 0]
            ])
        )
        return beta

    def _H(self, q):
        # `H` is independent of `q`.
        H = self.I
        return H

    # Build `_F_` on first use and then reuse.
    @functools.cached_property
    def _F_(self):
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
        # `F` is independent of `q`.
        F = self._F_
        return F

    _T_ = numpy.array([
        [0, 0, 0, 0, 0],
        [0, - 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    def _T(self, q):
        # `T` is independent of `q`.
        T = self._T_
        return T

    @staticmethod
    def _B():
        B = numpy.array([
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return B
