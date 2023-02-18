#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models.age_structured
import models._utility


def beta(solver):
    J = len(solver.model.ages)
    zeros = scipy.sparse.csr_array((1, J))
    ones = numpy.ones((1, J))
    return (solver.model.transmission.rate
            * solver.age_step
            * scipy.sparse.bmat([[zeros, zeros, zeros, ones, zeros]]))


def Hq(solver, q):
    J = len(solver.model.ages)
    if q == 0:
        HqXX = scipy.sparse.identity(J)
    elif q == 1:
        HqXX = models._utility.sparse.diags(
            {-1: numpy.ones(J - 1),
             0: numpy.hstack([numpy.zeros(J - 1), 1])})
    else:
        return ValueError
    return scipy.sparse.block_diag([HqXX] * 5)


def Fq(solver, q):
    J = len(solver.model.ages)

    def FqXW(pi):
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(J)
        if q == 0:
            return scipy.sparse.diags(pi)
        elif q == 1:
            return models._utility.sparse.diags(
                {-1: pi[:-1],
                 0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])})
        else:
            raise ValueError

    mu = solver.model.death.rate(solver.model.ages)
    omega = 1 / solver.model.waning.mean
    rho = 1 / solver.model.progression.mean
    gamma = 1 / solver.model.recovery.mean
    return scipy.sparse.bmat([
        [FqXW(- omega - mu), None, None, None, None],
        [FqXW(omega), FqXW(- mu), None, None, None],
        [None, None, FqXW(- rho - mu), None, None],
        [None, None, FqXW(rho), FqXW(- gamma - mu), None],
        [None, None, None, FqXW(gamma), FqXW(- mu)]
    ])


def Tq(solver, q):
    J = len(solver.model.ages)
    if q == 0:
        TqXW = scipy.sparse.identity(J)
    elif q == 1:
        TqXW = models._utility.sparse.diags(
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


def B(solver):
    J = len(solver.model.ages)
    BXW = scipy.sparse.lil_array((J, J))
    BXW[0] = solver.model.birth.maternity(solver.model.ages)
    Zeros = scipy.sparse.csr_array((J, J))
    return scipy.sparse.bmat([
        [Zeros, Zeros, Zeros, Zeros, BXW],
        [BXW, BXW, BXW, BXW, Zeros],
        [Zeros, Zeros, Zeros, Zeros, Zeros],
        [Zeros, Zeros, Zeros, Zeros, Zeros],
        [Zeros, Zeros, Zeros, Zeros, Zeros]
    ])


def check_matrices(solver):
    assert models._utility.sparse.equals(beta(solver), solver.beta)
    assert models._utility.sparse.equals(Hq(solver, 0), solver.H0)
    assert models._utility.sparse.equals(Hq(solver, 1), solver.H1)
    assert models._utility.sparse.equals(Fq(solver, 0), solver.F0)
    assert models._utility.sparse.equals(Fq(solver, 1), solver.F1)
    assert models._utility.sparse.equals(Tq(solver, 0), solver.T0)
    assert models._utility.sparse.equals(Tq(solver, 1), solver.T1)
    assert models._utility.sparse.equals(B(solver), solver.B)


if __name__ == '__main__':
    model = models.age_structured.Model()
    solver = model.solver()
    check_matrices(solver)
