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
        beta = (self.model.transmission.rate
                * numpy.array((0, 0, 0, 1, 0)))
        return beta

    def _H(self):
        H = self.I
        return H

    def _Hq(self, q):
        # `Hq` is independent of `q`.
        Hq = self._H()
        return Hq

    # Build `_F` on first use and then reuse.
    @functools.cached_property
    def _F(self):
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

    def _Fq(self, q):
        # `Fq` is independent of `q`.
        Fq = self._F
        return Fq

    _T = numpy.array([
        [0, 0, 0, 0, 0],
        [0, - 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    def _Tq(self, q):
        # `Tq` is independent of `q`.
        Tq = self._T
        return Tq

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
