#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models.age_structured
import models._utility

import _check_solver


class Checker(_check_solver.Base):
    def beta(self):
        J = len(self.model.a)
        zeros = scipy.sparse.csr_array((1, J))
        ones = numpy.ones((1, J))
        return (self.model.transmission.rate
                * self.model.a_step
                * scipy.sparse.bmat([[zeros, zeros, zeros, ones, zeros]]))

    def Hq(self, q):
        J = len(self.model.a)
        if q == 'new':
            HqXX = scipy.sparse.identity(J)
        elif q == 'cur':
            HqXX = models._utility.sparse.diags_from_dict(
                {-1: numpy.ones(J - 1),
                 0: numpy.hstack([numpy.zeros(J - 1), 1])})
        else:
            return ValueError
        return scipy.sparse.block_diag([HqXX] * 5)

    def Fq(self, q):
        J = len(self.model.a)

        def FqXW(pi):
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

        mu = self.model.death.rate(self.model.a)
        omega = 1 / self.model.waning.mean
        rho = 1 / self.model.progression.mean
        gamma = 1 / self.model.recovery.mean
        return scipy.sparse.bmat([
            [FqXW(- omega - mu), None, None, None, None],
            [FqXW(omega), FqXW(- mu), None, None, None],
            [None, None, FqXW(- rho - mu), None, None],
            [None, None, FqXW(rho), FqXW(- gamma - mu), None],
            [None, None, None, FqXW(gamma), FqXW(- mu)]
        ])

    def Tq(self, q):
        J = len(self.model.a)
        if q == 'new':
            TqXW = scipy.sparse.identity(J)
        elif q == 'cur':
            TqXW = models._utility.sparse.diags_from_dict(
                {-1: numpy.ones(J - 1),
                 0: numpy.hstack([numpy.zeros(J - 1), 1])})
        else:
            raise ValueError
        Zeros = scipy.sparse.csr_array((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, - TqXW, Zeros, Zeros, Zeros],
            [Zeros, TqXW, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])

    def B(self):
        J = len(self.model.a)
        BXW = scipy.sparse.lil_array((J, J))
        BXW[0] = self.model.birth.maternity(self.model.a)
        Zeros = scipy.sparse.csr_array((J, J))
        return scipy.sparse.bmat([
            [Zeros, Zeros, Zeros, Zeros, BXW],
            [BXW, BXW, BXW, BXW, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros],
            [Zeros, Zeros, Zeros, Zeros, Zeros]
        ])


if __name__ == '__main__':
    model = models.age_structured.Model()
    checker = Checker(model)
    checker.check_matrices()
