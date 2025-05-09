#!/usr/bin/python3
'''Check the solver matrices.'''

import numpy
import scipy.sparse

from context import models
from models import _utility

import solver_check


class Checker(solver_check.Checker):
    '''Check the age-structured solver.'''

    @property
    def J(self):
        return len(self.model.a)

    @property
    def iota(self):
        return (self.model.a_step
                * numpy.ones((1, self.J)))

    @property
    def zeta(self):
        zeta = scipy.sparse.dok_array((self.J, 1))
        zeta[0, 0] = 1 / self.model.a_step
        return zeta

    @property
    def L_XX(self):
        return _utility.sparse.diags_from_dict({
            -1: numpy.ones(self.J - 1),
            0: numpy.hstack([numpy.zeros(self.J - 1), 1]),
        })

    def H_XX(self, q):
        if q == 'new':
            H_XX = scipy.sparse.identity(self.J)
        elif q == 'cur':
            H_XX = self.L_XX
        else:
            return ValueError
        return H_XX

    def H(self, q):
        return scipy.sparse.block_diag([self.H_XX(q)] * 5)

    def F(self, q):
        def F_XW(pi):
            if numpy.isscalar(pi):
                pi = pi * numpy.ones(self.J)
            F_XW = scipy.sparse.diags(pi)
            if q == 'cur':
                F_XW = self.L_XX @ F_XW
            return F_XW

        def get_rate(which):
            waiting_time = getattr(self.model.parameters, which)
            rate = waiting_time.rate(self.model.a)
            return _utility.numerical.rate_make_finite(rate)

        mu = self.model.parameters.death.rate(self.model.a)
        omega = get_rate('waning')
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        return scipy.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_XW(- rho - mu), None, None],
            [None, None, F_XW(rho), F_XW(- gamma - mu), None],
            [None, None, None, F_XW(gamma), F_XW(- mu)]
        ])

    @property
    def tau(self):
        nu = self.model.parameters.birth.maternity(self.model.a)
        return (self.model.a_step * nu).reshape((1, -1))

    def B(self):
        B_XW = self.zeta @ self.tau
        Zeros = solver_check.Zeros((self.J, self.J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, B_XW],
            [B_XW, B_XW, B_XW, B_XW, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])

    def beta(self):
        zeros = solver_check.Zeros((1, self.J))
        return (self.model.parameters.transmission.rate
                * scipy.sparse.hstack(
                    [zeros, zeros, zeros, self.iota, zeros]
                ))

    def T(self, q):
        T_XW = self.H_XX(q)
        Zeros = solver_check.Zeros((self.J, self.J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, - T_XW, Zeros, Zeros, Zeros],
            [Zeros, T_XW, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])


if __name__ == '__main__':
    model = models.age_structured.Model()
    checker = Checker(model)
    checker.check()
