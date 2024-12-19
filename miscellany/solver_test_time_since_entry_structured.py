#!/usr/bin/python3
'''Test the solver matrices.'''

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Tester):
    '''Test the time-since-entry-structured solver.'''

    @property
    def K(self):
        return len(self.model.z)

    @property
    def iota(self):
        return (self.model.z_step
                * numpy.ones((1, self.K)))

    @property
    def zeta(self):
        zeta = scipy.sparse.dok_array((self.K, 1))
        zeta[0, 0] = 1 / self.model.z_step
        return zeta

    @property
    def L_yy(self):
        return models._utility.sparse.diags_from_dict({
            -1: numpy.ones(self.K - 1),
            0: numpy.hstack([numpy.zeros(self.K - 1), 1]),
        })

    def H(self, q):
        H_yy = scipy.sparse.identity(self.K)
        if q == 'cur':
            H_yy = self.L_yy @ H_yy
        H_XX = [[1]]
        return scipy.sparse.block_diag([H_yy, H_XX, H_yy, H_yy, H_XX])

    def sigma(self, xi):
        if numpy.isscalar(xi):
            xi = xi * numpy.ones(self.K)
        sigma = scipy.sparse.dok_array((1, self.K))
        sigma[0] = self.model.z_step * xi
        return sigma

    def F(self, q):
        def f_XX(pi):
            return [[pi]]

        def F_yy(xi):
            if numpy.isscalar(xi):
                xi = xi * numpy.ones(self.K)
            F_yy = scipy.sparse.diags(xi)
            if q == 'cur':
                F_yy = self.L_yy @ F_yy
            return F_yy

        def F_yw(xi):
            return self.zeta @ self.sigma(xi)

        def f_Xy(xi):
            return self.sigma(xi)

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

    def B(self):
        b_yX = self.zeta
        b_Xy = self.iota
        b_XX = [[1]]
        Zeros_yw = solver_test.Zeros((self.K, self.K))
        zeros_yX = solver_test.Zeros((self.K, 1))
        zeros_Xy = solver_test.Zeros((1, self.K))
        zeros_XW = solver_test.Zeros((1, 1))
        return scipy.sparse.bmat([
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, b_yX],
            [b_Xy, b_XX, b_Xy, b_Xy, zeros_XW],
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_Xy, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW]
        ])

    def beta(self):
        zeros_X = solver_test.Zeros((1, 1))
        zeros_y = solver_test.Zeros((1, self.K))
        return (self.model.parameters.transmission.rate
                * scipy.sparse.hstack(
                    [zeros_y, zeros_X, zeros_y, self.iota, zeros_X]
                ))

    def T(self, q):
        # `T` is independent of `q`.
        t_XX = numpy.array([[1]])
        t_yX = self.zeta
        Zeros_yw = solver_test.Zeros((self.K, self.K))
        zeros_yX = solver_test.Zeros((self.K, 1))
        zeros_Xy = solver_test.Zeros((1, self.K))
        zeros_XW = solver_test.Zeros((1, 1))
        return scipy.sparse.bmat([
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_Xy, - t_XX, zeros_Xy, zeros_Xy, zeros_XW],
            [Zeros_yw, t_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [Zeros_yw, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_Xy, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW]
        ])


if __name__ == '__main__':
    model = models.time_since_entry_structured.Model()
    tester = Tester(model)
    tester.test()
