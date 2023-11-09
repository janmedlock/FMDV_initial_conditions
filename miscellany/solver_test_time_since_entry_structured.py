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
        zeros_X = solver_test.SparseArray((1, 1))
        zeros_y = solver_test.SparseArray((1, K))
        iota_y = (self.model.z_step
                  * numpy.ones((1, K)))
        return (self.model.parameters.transmission.rate
                * scipy.sparse.hstack(
                    [zeros_y, zeros_X, zeros_y, iota_y, zeros_X]
                ))

    def H(self, q):
        K = len(self.model.z)
        if q == 'new':
            H_yy = scipy.sparse.identity(K)
        elif q == 'cur':
            H_yy = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(K - 1),
                0: numpy.hstack([numpy.zeros(K - 1), 1])
            })
        else:
            return ValueError
        H_XX = [[1]]
        return scipy.sparse.block_diag([H_yy, H_XX, H_yy, H_yy, H_XX])

    def F(self, q):
        K = len(self.model.z)

        def F_yy(xi):
            if numpy.isscalar(xi):
                xi = xi * numpy.ones(K)
            if q == 'new':
                return scipy.sparse.diags(xi)
            elif q == 'cur':
                return models._utility.sparse.diags_from_dict({
                    -1: xi[:-1],
                    0: numpy.hstack([numpy.zeros(K - 1), xi[-1]])
                })
            else:
                raise ValueError

        def F_yw(xi):
            if numpy.isscalar(xi):
                xi = xi * numpy.ones(K)
            F_yw_ = scipy.sparse.dok_array((K, K))
            F_yw_[0] = xi
            return F_yw_

        def f_Xy(xi):
            if numpy.isscalar(xi):
                xi = xi * numpy.ones(K)
            f_Xy_ = scipy.sparse.dok_array((1, K))
            f_Xy_[0] = self.model.z_step * xi
            return f_Xy_

        def f_XX(pi):
            return [[pi]]

        def get_rate(which):
            waiting_time = getattr(self.model.parameters, which)
            rate = waiting_time.rate(self.model.z)
            return models._utility.numerical.rate_make_finite(rate)

        mu = self.model.parameters.death_rate_mean
        omega = get_rate('waning')
        rho = get_rate('progression')
        gamma = get_rate('recovery')
        return scipy.sparse.bmat([
            [F_yy(- omega - mu), None, None, None, None],
            [f_Xy(omega), f_XX(- mu), None, None, None],
            [None, None, F_yy(- rho - mu), None, None],
            [None, None, F_yw(rho), F_yy(- gamma - mu), None],
            [None, None, None, f_Xy(gamma), f_XX(- mu)]
        ])

    def T(self, q):
        # `T` is independent of `q`.
        K = len(self.model.z)
        t_XX = numpy.array([[1]])
        t_yX = scipy.sparse.dok_array((K, 1))
        t_yX[0, 0] = 1 / self.model.z_step
        Zeros_yw = solver_test.SparseArray((K, K))
        zeros_yX = solver_test.SparseArray((K, 1))
        zeros_Xy = solver_test.SparseArray((1, K))
        zeros_XW = solver_test.SparseArray((1, 1))
        return scipy.sparse.bmat([
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_Xy, - t_XX, zeros_Xy, zeros_Xy, zeros_XW],
            [Zeros_yw, t_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_Xy, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW]
        ])

    def B(self):
        K = len(self.model.z)
        b_yX = scipy.sparse.dok_array((K, 1))
        b_yX[0, 0] = 1 / self.model.z_step
        b_Xy = scipy.sparse.dok_array((1, K))
        b_Xy[0] = self.model.z_step
        b_XX = [[1]]
        Zeros_yw = solver_test.SparseArray((K, K))
        zeros_yX = solver_test.SparseArray((K, 1))
        zeros_Xy = solver_test.SparseArray((1, K))
        zeros_XW = solver_test.SparseArray((1, 1))
        return scipy.sparse.bmat([
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, b_yX],
            [b_Xy, b_XX, b_Xy, b_Xy, zeros_XW],
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_Xy, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW]
        ])


if __name__ == '__main__':
    model = models.time_since_entry_structured.Model()
    tester = Tester(model)
    tester.test()
