#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Tester):
    '''Test the age– and time-since-entry–structured solver.'''

    def beta(self):
        J = len(self.model.a)
        K = len(self.model.z)
        zeros_X = solver_test.SparseArray((1, J))
        zeros_y = solver_test.SparseArray((1, J * K))
        iota_y = (self.model.a_step
                  * self.model.z_step
                  * numpy.ones((1, J * K)))
        return (self.model.parameters.transmission.rate
                * scipy.sparse.hstack(
                    [zeros_X, zeros_X, zeros_y, iota_y, zeros_X]
                ))

    def H(self, q):
        J = len(self.model.a)
        K = len(self.model.z)
        if q == 'new':
            H_a = scipy.sparse.identity(J)
            H_z = scipy.sparse.identity(K)
        elif q == 'cur':
            H_a = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            })
            H_z = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(K - 1),
                0: numpy.hstack([numpy.zeros(K - 1), 1])
            })
        else:
            return ValueError
        H_XX = H_a
        H_yy = scipy.sparse.kron(H_a, H_z)
        return scipy.sparse.block_diag([H_XX, H_XX, H_yy, H_yy, H_XX])

    def F(self, q):
        J = len(self.model.a)
        K = len(self.model.z)
        if q == 'new':
            H_a = scipy.sparse.identity(J)
            H_z = scipy.sparse.identity(K)
        elif q == 'cur':
            H_a = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            })
            H_z = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(K - 1),
                0: numpy.hstack([numpy.zeros(K - 1), 1])
            })
        else:
            return ValueError
        zeta_z = scipy.sparse.dok_array((K, 1))
        zeta_z[0, 0] = 1 / self.model.z_step

        def F_a(pi):
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

        def F_z(xi):
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

        def F_XW(pi):
            return F_a(pi)

        def F_yy(pi, xi):
            return (scipy.sparse.kron(F_a(pi), H_z)
                    + scipy.sparse.kron(H_a, F_z(xi)))

        def sigma_z(xi):
            if numpy.isscalar(xi):
                xi = xi * numpy.ones(K)
            return (self.model.z_step * xi.reshape((1, K)))

        def F_Xy(xi):
            return scipy.sparse.kron(H_a, sigma_z(xi))

        def F_yw(xi):
            return scipy.sparse.kron(H_a, zeta_z @ sigma_z(xi))

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

    def T(self, q):
        J = len(self.model.a)
        K = len(self.model.z)
        if q == 'new':
            T_a = scipy.sparse.identity(J)
        elif q == 'cur':
            T_a = models._utility.sparse.diags_from_dict({
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            })
        else:
            return ValueError
        zeta_z = scipy.sparse.dok_array((K, 1))
        zeta_z[0, 0] = 1 / self.model.z_step
        T_XX = T_a
        T_yX = scipy.sparse.kron(T_a, zeta_z)
        Zeros_yw = solver_test.SparseArray((J * K, J * K))
        Zeros_yX = solver_test.SparseArray((J * K, J))
        Zeros_Xy = solver_test.SparseArray((J, J * K))
        Zeros_XW = solver_test.SparseArray((J, J))
        return scipy.sparse.bmat([
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW],
            [Zeros_XW, - T_XX, Zeros_Xy, Zeros_Xy, Zeros_XW],
            [Zeros_yX, T_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW]
        ])

    def B(self):
        J = len(self.model.a)
        K = len(self.model.z)
        b_a = scipy.sparse.dok_array((J, 1))
        b_a[0, 0] = 1 / self.model.a_step
        nu = self.model.parameters.birth.maternity(self.model.a)
        tau_a = solver_test.SparseArray(self.model.a_step
                                        * nu.reshape((1, J)))
        tau_z = solver_test.SparseArray(self.model.z_step
                                        * numpy.ones((1, K)))
        tau_y = scipy.sparse.kron(tau_a, tau_z)
        B_XW = b_a @ tau_a
        B_Xy = b_a @ tau_y
        Zeros_yw = solver_test.SparseArray((J * K, J * K))
        Zeros_yX = solver_test.SparseArray((J * K, J))
        Zeros_Xy = solver_test.SparseArray((J, J * K))
        Zeros_XW = solver_test.SparseArray((J, J))
        return scipy.sparse.bmat([
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, B_XW],
            [B_XW, B_XW, B_Xy, B_Xy, Zeros_XW],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW]
        ])


if __name__ == '__main__':
    model = models.combination.Model()
    tester = Tester(model)
    tester.test()
