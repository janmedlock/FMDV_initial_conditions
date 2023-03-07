#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Base):
    def beta(self):
        J = len(self.model.a)
        zeros = solver_test.sparse_array((1, J))
        ones = numpy.ones((1, J))
        return (self.model.parameters.transmission.rate
                * self.model.a_step
                * scipy.sparse.hstack(
                    [zeros, zeros, zeros, ones, zeros]
                ))

    def H(self, q):
        J = len(self.model.a)
        if q == 'new':
            HXX = scipy.sparse.identity(J)
        elif q == 'cur':
            HXX = models._utility.sparse.diags_from_dict(
                {-1: numpy.ones(J - 1),
                 0: numpy.hstack([numpy.zeros(J - 1), 1])})
        else:
            return ValueError
        return scipy.sparse.block_diag([HXX] * 5)

    def F(self, q):
        J = len(self.model.a)

        def FXW(pi):
            if numpy.isscalar(pi):
                pi = pi * numpy.ones(J)
            if q == 'new':
                return scipy.sparse.diags(pi)
            elif q == 'cur':
                return models._utility.sparse.diags_from_dict(
                    {-1: pi[:-1],
                     0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])})
            else:
                raise ValueError

        mu = self.model.parameters.death.rate(self.model.a)
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        return scipy.sparse.bmat([
            [FXW(- omega - mu), None, None, None, None],
            [FXW(omega), FXW(- mu), None, None, None],
            [None, None, FXW(- rho - mu), None, None],
            [None, None, FXW(rho), FXW(- gamma - mu), None],
            [None, None, None, FXW(gamma), FXW(- mu)]
        ])

    def T(self, q):
        J = len(self.model.a)
        if q == 'new':
            TXW = scipy.sparse.identity(J)
        elif q == 'cur':
            TXW = models._utility.sparse.diags_from_dict(
                {-1: numpy.ones(J - 1),
                 0: numpy.hstack([numpy.zeros(J - 1), 1])})
        else:
            raise ValueError
        Zeros = solver_test.sparse_array((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, - TXW, Zeros, Zeros, Zeros],
            [Zeros, TXW, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])

    def B(self):
        J = len(self.model.a)
        nu = self.model.parameters.birth.maternity(self.model.a)
        BXW = scipy.sparse.dok_array((J, J))
        BXW[0] = nu
        Zeros = solver_test.sparse_array((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, BXW],
            [BXW, BXW, BXW, BXW, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])


if __name__ == '__main__':
    model = models.age_structured.Model()
    tester = Tester(model)
    tester.test()
