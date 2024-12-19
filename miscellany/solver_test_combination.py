#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Tester):
    '''Test the age– and time-since-entry–structured solver.'''

    @property
    def J(self):
        return len(self.model.a)

    @property
    def K(self):
        return len(self.model.z)

    @property
    def iota_a(self):
        return (self.model.a_step
                * numpy.ones((1, self.J)))

    @property
    def iota_z(self):
        return (self.model.z_step
                * numpy.ones((1, self.K)))

    @property
    def zeta_a(self):
        zeta_a = scipy.sparse.dok_array((self.J, 1))
        zeta_a[0, 0] = 1 / self.model.a_step
        return zeta_a

    @property
    def zeta_z(self):
        zeta_z = scipy.sparse.dok_array((self.K, 1))
        zeta_z[0, 0] = 1 / self.model.z_step
        return zeta_z

    @property
    def L_a(self):
        return models._utility.sparse.diags_from_dict({
            -1: numpy.ones(self.J - 1),
            0: numpy.hstack([numpy.zeros(self.J - 1), 1]),
        })

    @property
    def L_z(self):
        return models._utility.sparse.diags_from_dict({
            -1: numpy.ones(self.K - 1),
            0: numpy.hstack([numpy.zeros(self.K - 1), 1]),
        })

    def H_a(self, q):
        H_a = scipy.sparse.identity(self.J)
        if q == 'cur':
            H_a = self.L_a @ H_a
        return H_a

    def H_z(self, q):
        H_z = scipy.sparse.identity(self.K)
        if q == 'cur':
            H_z = self.L_z @ H_z
        return H_z

    def H(self, q):
        H_a = self.H_a(q)
        H_z = self.H_z(q)
        H_XX = H_a
        H_yy = scipy.sparse.kron(H_a, H_z)
        return scipy.sparse.block_diag([H_XX, H_XX, H_yy, H_yy, H_XX])

    def sigma_z(self, xi):
        if numpy.isscalar(xi):
            xi = xi * numpy.ones(self.K)
        sigma_z = scipy.sparse.dok_array((1, self.K))
        sigma_z[0] = self.model.z_step * xi
        return sigma_z

    def F(self, q):
        H_a = self.H_a(q)
        H_z = self.H_z(q)

        def F_a(pi):
            if numpy.isscalar(pi):
                pi = pi * numpy.ones(self.J)
            F_a = scipy.sparse.diags(pi)
            if q == 'cur':
                F_a = self.L_a @ F_a
            return F_a

        def F_z(xi):
            if numpy.isscalar(xi):
                xi = xi * numpy.ones(self.K)
            F_z = scipy.sparse.diags(xi)
            if q == 'cur':
                F_z = self.L_z @ F_z
            return F_z

        def F_XW(pi):
            return F_a(pi)

        def F_yy(pi, xi):
            return (scipy.sparse.kron(F_a(pi), H_z)
                    + scipy.sparse.kron(H_a, F_z(xi)))

        def F_Xy(xi):
            return scipy.sparse.kron(H_a, self.sigma_z(xi))

        def F_yw(xi):
            return scipy.sparse.kron(H_a, self.zeta_z @ self.sigma_z(xi))

        def get_rate_a(which):
            waiting_time = getattr(self.model.parameters, which)
            rate = waiting_time.rate(self.model.a)
            return models._utility.numerical.rate_make_finite(rate)

        def get_rate_z(which):
            waiting_time = getattr(self.model.parameters, which)
            rate = waiting_time.rate(self.model.z)
            return models._utility.numerical.rate_make_finite(rate)

        mu = get_rate_a('death')
        omega = get_rate_a('waning')
        rho = get_rate_z('progression')
        gamma = get_rate_z('recovery')
        return scipy.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_yy(- mu, - rho), None, None],
            [None, None, F_yw(rho), F_yy(- mu, - gamma), None],
            [None, None, None, F_Xy(gamma), F_XW(- mu)]
        ])

    @property
    def tau_a(self):
        nu = self.model.parameters.birth.maternity(self.model.a)
        return (self.model.a_step * nu).reshape((1, -1))

    def B(self):
        B_XW = self.zeta_a @ self.tau_a
        B_Xy = self.zeta_a @ scipy.sparse.kron(self.tau_a, self.iota_z)
        Zeros_yw = solver_test.Zeros((self.J * self.K, self.J * self.K))
        Zeros_yX = solver_test.Zeros((self.J * self.K, self.J))
        Zeros_Xy = solver_test.Zeros((self.J, self.J * self.K))
        Zeros_XW = solver_test.Zeros((self.J, self.J))
        return scipy.sparse.bmat([
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, B_XW],
            [B_XW,     B_XW,     B_Xy,     B_Xy,     Zeros_XW],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW]
        ])

    def beta(self):
        iota_y = scipy.sparse.kron(self.iota_a, self.iota_z)
        zeros_X = solver_test.Zeros((1, self.J))
        zeros_y = solver_test.Zeros((1, self.J * self.K))
        return (self.model.parameters.transmission.rate
                * scipy.sparse.hstack(
                    [zeros_X, zeros_X, zeros_y, iota_y, zeros_X]
                ))

    def T(self, q):
        T_a = self.H_a(q)
        T_XX = T_a
        T_yX = scipy.sparse.kron(T_a, self.zeta_z)
        Zeros_yw = solver_test.Zeros((self.J * self.K, self.J * self.K))
        Zeros_yX = solver_test.Zeros((self.J * self.K, self.J))
        Zeros_Xy = solver_test.Zeros((self.J, self.J * self.K))
        Zeros_XW = solver_test.Zeros((self.J, self.J))
        return scipy.sparse.bmat([
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW],
            [Zeros_XW, - T_XX,   Zeros_Xy, Zeros_Xy, Zeros_XW],
            [Zeros_yX, T_yX,     Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW]
        ])


if __name__ == '__main__':
    model = models.combination.Model()
    tester = Tester(model)
    tester.test()
