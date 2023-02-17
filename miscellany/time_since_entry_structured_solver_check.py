#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models.time_since_entry_structured
import models._utility


def sparse_equals(a, b):
    return (a.shape == b.shape) & ((a != b).nnz == 0)


def beta(solver):
    K = len(solver.model.z)
    zeros = scipy.sparse.csr_array((1, K))
    ones = numpy.ones((1, K))
    return (solver.model.transmission.rate
            * solver.z_step
            * scipy.sparse.bmat([[zeros, [[0]], zeros, ones, [[0]]]]))


def Hq(solver, q):
    K = len(solver.model.z)
    if q == 0:
        Hqyy = scipy.sparse.identity(K)
    elif q == 1:
        Hqyy = models._utility.sparse.diags(
            {-1: numpy.ones(K - 1),
             0: numpy.hstack([numpy.zeros(K - 1), 1])})
    else:
        return ValueError
    return scipy.sparse.block_diag([Hqyy, [[1]], Hqyy, Hqyy, [[1]]])


def Fq(solver, q):
    K = len(solver.model.z)

    def Fqyy(psi):
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        if q == 0:
            return scipy.sparse.diags(psi)
        elif q == 1:
            return models._utility.sparse.diags(
                {-1: psi[:-1],
                 0: numpy.hstack([numpy.zeros(K - 1), psi[-1]])})
        else:
            raise ValueError

    def Fyz(psi):
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        Fyz_ = scipy.sparse.lil_array((K, K))
        Fyz_[0] = psi
        return Fyz_

    def fXy(psi):
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        fXy_ = scipy.sparse.lil_array((1, K))
        fXy_[0] = solver.z_step * psi
        return fXy_

    def get_rate(which):
        param = getattr(solver.model, which)
        rate = param.rate(solver.model.z)
        return models._utility.rate_make_finite(rate)

    mu = solver.model.death_rate_mean
    omega = get_rate('waning')
    rho = get_rate('progression')
    gamma = get_rate('recovery')
    return scipy.sparse.bmat([
        [Fqyy(- omega - mu), None, None, None, None],
        [fXy(omega), [[- mu]], None, None, None],
        [None, None, Fqyy(- rho - mu), None, None],
        [None, None, Fyz(rho), Fqyy(- gamma - mu), None],
        [None, None, None, fXy(gamma), [[- mu]]]
    ])


def T(solver):
    K = len(solver.model.z)
    tyX = scipy.sparse.lil_array((K, 1))
    tyX[0, 0] = 1 / solver.z_step
    ZerosKK = scipy.sparse.csr_array((K, K))
    zerosK1 = scipy.sparse.csr_array((K, 1))
    zeros1K = scipy.sparse.csr_array((1, K))
    return scipy.sparse.bmat([
        [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
        [zeros1K, [[- 1]], zeros1K, zeros1K, [[0]]],
        [ZerosKK, tyX, ZerosKK, ZerosKK, zerosK1],
        [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
        [zeros1K, [[0]], zeros1K, zeros1K, [[0]]]
    ])


def B(solver):
    K = len(solver.model.z)
    byX = scipy.sparse.lil_array((K, 1))
    byX[0, 0] = 1 / solver.z_step
    bXy = scipy.sparse.lil_array((1, K))
    bXy[0] = solver.z_step
    ZerosKK = scipy.sparse.csr_array((K, K))
    zerosK1 = scipy.sparse.csr_array((K, 1))
    zeros1K = scipy.sparse.csr_array((1, K))
    return scipy.sparse.bmat([
        [ZerosKK, zerosK1, ZerosKK, ZerosKK, byX],
        [bXy, [[1]], bXy, bXy, [[0]]],
        [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
        [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
        [zeros1K, [[0]], zeros1K, zeros1K, [[0]]]
    ])


def check_matrices(solver):
    assert sparse_equals(beta(solver), solver.beta)
    assert sparse_equals(Hq(solver, 0), solver.H0)
    assert sparse_equals(Hq(solver, 1), solver.H1)
    assert sparse_equals(Fq(solver, 0), solver.F0)
    assert sparse_equals(Fq(solver, 1), solver.F1)
    assert sparse_equals(T(solver), solver.T)
    assert sparse_equals(B(solver), solver.B)


if __name__ == '__main__':
    model = models.time_since_entry_structured.Model()
    solver = model.solver()
    check_matrices(solver)
