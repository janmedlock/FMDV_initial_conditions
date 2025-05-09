#!/usr/bin/python3
'''Test the solver matrices for the age-structured model.'''

import functools

import numpy
import scipy.sparse

from context import models

import solver


class TestSolver(solver.TestSolver):
    '''Test the age-structured solver.'''

    Model = models.age_structured.Model

    @functools.cache
    def J(self, model):
        return len(model.a)

    @functools.cache
    def iota(self, model):
        return (model.a_step * numpy.ones((1, self.J(model))))

    @functools.cache
    def zeta(self, model):
        zeta = scipy.sparse.dok_array((self.J(model), 1))
        zeta[0, 0] = 1 / model.a_step
        return zeta

    @functools.cache
    def L_XX(self, model):
        J = self.J(model)
        return solver.sparse.diags_from_dict({
            -1: numpy.ones(J - 1),
            0: numpy.hstack([numpy.zeros(J - 1), 1]),
        })

    @functools.cache
    def H_XX(self, model, q):
        if q == 'new':
            H_XX = scipy.sparse.identity(self.J(model))
        elif q == 'cur':
            H_XX = self.L_XX(model)
        else:
            return ValueError
        return H_XX

    def H(self, model, q):
        return scipy.sparse.block_diag([self.H_XX(model, q)] * 5)

    def F_XW(self, model, q, pi):
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(self.J(model))
        F_XW = scipy.sparse.diags(pi)
        if q == 'cur':
            F_XW = self.L_XX(model) @ F_XW
        return F_XW

    @functools.cache
    def _get_rate(self, model, which):
        waiting_time = getattr(model.parameters, which)
        rate = waiting_time.rate(model.a)
        return solver.numerical.rate_make_finite(rate)

    def F(self, model, q):
        F_XW = functools.partial(self.F_XW, model, q)
        mu = model.parameters.death.rate(model.a)
        omega = self._get_rate(model, 'waning')
        rho = 1 / model.parameters.progression.mean
        gamma = 1 / model.parameters.recovery.mean
        return scipy.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_XW(- rho - mu), None, None],
            [None, None, F_XW(rho), F_XW(- gamma - mu), None],
            [None, None, None, F_XW(gamma), F_XW(- mu)]
        ])

    @functools.cache
    def tau(self, model):
        nu = model.parameters.birth.maternity(model.a)
        return (model.a_step * nu).reshape((1, -1))

    def B(self, model):
        B_XW = self.zeta(model) @ self.tau(model)
        J = self.J(model)
        Zeros = solver.Zeros((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, B_XW],
            [B_XW, B_XW, B_XW, B_XW, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])

    def beta(self, model):
        iota = self.iota(model)
        zeros = solver.Zeros((1, self.J(model)))
        return (
            model.parameters.transmission.rate
            * scipy.sparse.hstack(
                [zeros, zeros, zeros, iota, zeros]
            )
        )

    def T(self, model, q):
        T_XW = self.H_XX(model, q)
        J = self.J(model)
        Zeros = solver.Zeros((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, - T_XW, Zeros, Zeros, Zeros],
            [Zeros, T_XW, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])
