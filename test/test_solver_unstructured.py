#!/usr/bin/python3
'''Test the solver matrices for the unstructured model.'''

import functools

import numpy

from context import models

import solver


class TestSolver(solver.TestSolver):
    '''Test the unstructured solver.'''

    Model = models.unstructured.Model

    @functools.cache
    def _H(self, model):
        # `H` is independent of `q`.
        return numpy.identity(5)

    def H(self, model, q):
        return self._H(model)

    @functools.cache
    def _F(self, model):
        # `F` is independent of `q`.
        mu = model.parameters.death_rate_mean
        omega = 1 / model.parameters.waning.mean
        rho = 1 / model.parameters.progression.mean
        gamma = 1 / model.parameters.recovery.mean
        return numpy.array([
            [- omega - mu, 0, 0, 0, 0],
            [omega, - mu, 0, 0, 0],
            [0, 0, - rho - mu, 0, 0],
            [0, 0, rho, - gamma - mu, 0],
            [0, 0, 0, gamma, - mu]
        ])

    def F(self, model, q):
        return self._F(model)

    def B(self, model):
        return numpy.array([
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

    def beta(self, model):
        return (
            model.parameters.transmission.rate
            * numpy.array(
                [0, 0, 0, 1, 0]
            ).reshape((1, -1))
        )

    @functools.cache
    def _T(self, model):
        # `T` is independent of `q`.
        return numpy.array([
            [0,   0, 0, 0, 0],
            [0, - 1, 0, 0, 0],
            [0,   1, 0, 0, 0],
            [0,   0, 0, 0, 0],
            [0,   0, 0, 0, 0]
        ])

    def T(self, model, q):
        return self._T(model)
