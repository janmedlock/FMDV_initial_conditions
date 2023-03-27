#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Tester):
    '''Test the time-since-entry-structured solver.'''

    def beta(self):
        K = len(self.model.z)
        zeros = solver_test.sparse_array((1, K))
        ones = numpy.ones((1, K))
        return (self.model.parameters.transmission.rate
                * self.model.z_step
                * scipy.sparse.hstack(
                    [zeros, [[0]], zeros, ones, [[0]]]
                ))

    def H(self, q):
        K = len(self.model.z)
        if q == 'new':
            Hqyy = scipy.sparse.identity(K)
        elif q == 'cur':
            Hqyy = models._utility.sparse.diags_from_dict(
                {-1: numpy.ones(K - 1),
                 0: numpy.hstack([numpy.zeros(K - 1), 1])})
        else:
            return ValueError
        return scipy.sparse.block_diag([Hqyy, [[1]], Hqyy, Hqyy, [[1]]])

    def F(self, q):
        K = len(self.model.z)

        def Fqyy(psi):
            if numpy.isscalar(psi):
                psi = psi * numpy.ones(K)
            if q == 'new':
                return scipy.sparse.diags(psi)
            elif q == 'cur':
                return models._utility.sparse.diags_from_dict(
                    {-1: psi[:-1],
                     0: numpy.hstack([numpy.zeros(K - 1), psi[-1]])})
            else:
                raise ValueError

        def Fyz(psi):
            if numpy.isscalar(psi):
                psi = psi * numpy.ones(K)
            Fyz_ = scipy.sparse.dok_array((K, K))
            Fyz_[0] = psi
            return Fyz_

        def fXy(psi):
            if numpy.isscalar(psi):
                psi = psi * numpy.ones(K)
            fXy_ = scipy.sparse.dok_array((1, K))
            fXy_[0] = self.model.z_step * psi
            return fXy_

        def get_rate(which):
            param = getattr(self.model.parameters, which)
            rate = param.rate(self.model.z)
            return models._utility.numerical.rate_make_finite(rate)

        mu = self.model.parameters.death_rate_mean
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

    def T(self, q):
        # `T` is independent of `q`.
        K = len(self.model.z)
        tyX = scipy.sparse.dok_array((K, 1))
        tyX[0, 0] = 1 / self.model.z_step
        ZerosKK = solver_test.sparse_array((K, K))
        zerosK1 = solver_test.sparse_array((K, 1))
        zeros1K = solver_test.sparse_array((1, K))
        return scipy.sparse.bmat([
            [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
            [zeros1K, [[- 1]], zeros1K, zeros1K, [[0]]],
            [ZerosKK, tyX, ZerosKK, ZerosKK, zerosK1],
            [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
            [zeros1K, [[0]], zeros1K, zeros1K, [[0]]]
        ])

    def B(self):
        K = len(self.model.z)
        byX = scipy.sparse.dok_array((K, 1))
        byX[0, 0] = 1 / self.model.z_step
        bXy = scipy.sparse.dok_array((1, K))
        bXy[0] = self.model.z_step
        ZerosKK = solver_test.sparse_array((K, K))
        zerosK1 = solver_test.sparse_array((K, 1))
        zeros1K = solver_test.sparse_array((1, K))
        return scipy.sparse.bmat([
            [ZerosKK, zerosK1, ZerosKK, ZerosKK, byX],
            [bXy, [[1]], bXy, bXy, [[0]]],
            [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
            [ZerosKK, zerosK1, ZerosKK, ZerosKK, zerosK1],
            [zeros1K, [[0]], zeros1K, zeros1K, [[0]]]
        ])


if __name__ == '__main__':
    model = models.time_since_entry_structured.Model()
    tester = Tester(model)
    tester.test()
