#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Tester):
    '''Test the age-structured solver.'''

    def beta(self):
        J = len(self.model.a)
        zeros = solver_test.sparse_array((1, J))
        iota = (self.model.a_step
                * numpy.ones((1, J)))
        return (self.model.parameters.transmission.rate
                * scipy.sparse.hstack(
                    [zeros, zeros, zeros, iota, zeros]
                ))

    def H(self, q):
        J = len(self.model.a)
        if q == 'new':
            H_XX = scipy.sparse.identity(J)
        elif q == 'cur':
            H_XX = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            })
        else:
            return ValueError
        return scipy.sparse.block_diag([H_XX] * 5)

    def F(self, q):
        J = len(self.model.a)

        def F_XW(pi):
            if numpy.isscalar(pi):
                pi = pi * numpy.ones(J)
            if q == 'new':
                return scipy.sparse.diags(pi)
            elif q == 'cur':
                return models._utility.sparse.diags_from_dict({
                    -1: pi[:-1],
                    0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])
                })
            else:
                raise ValueError

        mu = self.model.parameters.death.rate(self.model.a)
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        return scipy.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_XW(- rho - mu), None, None],
            [None, None, F_XW(rho), F_XW(- gamma - mu), None],
            [None, None, None, F_XW(gamma), F_XW(- mu)]
        ])

    def T(self, q):
        J = len(self.model.a)
        if q == 'new':
            T_XW = scipy.sparse.identity(J)
        elif q == 'cur':
            T_XW = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            })
        else:
            raise ValueError
        Zeros = solver_test.sparse_array((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, - T_XW, Zeros, Zeros, Zeros],
            [Zeros, T_XW, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])

    def B(self):
        J = len(self.model.a)
        nu = self.model.parameters.birth.maternity(self.model.a)
        B_XW = scipy.sparse.dok_array((J, J))
        B_XW[0] = nu
        Zeros = solver_test.sparse_array((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, B_XW],
            [B_XW, B_XW, B_XW, B_XW, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])


if __name__ == '__main__':
    model = models.age_structured.Model()
    tester = Tester(model)
    tester.test()
