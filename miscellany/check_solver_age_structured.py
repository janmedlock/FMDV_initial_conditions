#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models.age_structured
import models._utility


def beta(solver):
    J = len(solver.model.a)
    zeros = scipy.sparse.csr_array((1, J))
    ones = numpy.ones((1, J))
    return (solver.model.transmission.rate
            * solver.a_step
            * scipy.sparse.bmat([[zeros, zeros, zeros, ones, zeros]]))


def Hq(solver, q):
    J = len(solver.model.a)
    if q == 'new':
        HqXX = scipy.sparse.identity(J)
    elif q == 'cur':
        HqXX = models._utility.sparse.diags_from_dict(
            {-1: numpy.ones(J - 1),
             0: numpy.hstack([numpy.zeros(J - 1), 1])})
    else:
        return ValueError
    return scipy.sparse.block_diag([HqXX] * 5)


def Fq(solver, q):
    J = len(solver.model.a)

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

    mu = solver.model.death.rate(solver.model.a)
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
    J = len(solver.model.a)
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


def B(solver):
    J = len(solver.model.a)
    BXW = scipy.sparse.lil_array((J, J))
    BXW[0] = solver.model.birth.maternity(solver.model.a)
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
    assert models._utility.sparse.equals(Hq(solver, 'new'), solver.H_new)
    assert models._utility.sparse.equals(Hq(solver, 'cur'), solver.H_cur)
    assert models._utility.sparse.equals(Fq(solver, 'new'), solver.F_new)
    assert models._utility.sparse.equals(Fq(solver, 'cur'), solver.F_cur)
    assert models._utility.sparse.equals(Tq(solver, 'new'), solver.T_new)
    assert models._utility.sparse.equals(Tq(solver, 'cur'), solver.T_cur)
    assert models._utility.sparse.equals(B(solver), solver.B)


if __name__ == '__main__':
    model = models.age_structured.Model()
    check_matrices(model._solver)
