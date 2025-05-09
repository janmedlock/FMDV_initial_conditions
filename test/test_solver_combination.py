#!/usr/bin/python3
'''Test the solver matrices for the combination model.'''

import functools

import numpy
import scipy.sparse

from context import models

import solver


class TestSolver(solver.TestSolver):
    '''Test the age– and time-since-entry–structured solver.'''

    Model = models.combination.Model

    @functools.cache
    def J(self, model):
        return len(model.a)

    @functools.cache
    def K(self, model):
        return len(model.z)

    @functools.cache
    def iota_a(self, model):
        return (model.a_step * numpy.ones((1, self.J(model))))

    @functools.cache
    def iota_z(self, model):
        return (model.z_step * numpy.ones((1, self.K(model))))

    @functools.cache
    def zeta_a(self, model):
        zeta_a = scipy.sparse.dok_array((self.J(model), 1))
        zeta_a[0, 0] = 1 / model.a_step
        return zeta_a

    @functools.cache
    def zeta_z(self, model):
        zeta_z = scipy.sparse.dok_array((self.K(model), 1))
        zeta_z[0, 0] = 1 / model.z_step
        return zeta_z

    @functools.cache
    def L_a(self, model):
        J = self.J(model)
        return solver.sparse.diags_from_dict({
            -1: numpy.ones(J - 1),
            0: numpy.hstack([numpy.zeros(J - 1), 1]),
        })

    @functools.cache
    def L_z(self, model):
        K = self.K(model)
        return solver.sparse.diags_from_dict({
            -1: numpy.ones(K - 1),
            0: numpy.hstack([numpy.zeros(K - 1), 1]),
        })

    @functools.cache
    def H_a(self, model, q):
        H_a = scipy.sparse.identity(self.J(model))
        if q == 'cur':
            H_a = self.L_a(model) @ H_a
        return H_a

    @functools.cache
    def H_z(self, model, q):
        H_z = scipy.sparse.identity(self.K(model))
        if q == 'cur':
            H_z = self.L_z(model) @ H_z
        return H_z

    def H(self, model, q):
        H_a = self.H_a(model, q)
        H_z = self.H_z(model, q)
        H_XX = H_a
        H_yy = scipy.sparse.kron(H_a, H_z)
        return scipy.sparse.block_diag([H_XX, H_XX, H_yy, H_yy, H_XX])

    def sigma_z(self, model, xi):
        K = self.K(model)
        if numpy.isscalar(xi):
            xi = xi * numpy.ones(K)
        sigma_z = scipy.sparse.dok_array((1, K))
        sigma_z[0] = model.z_step * xi
        return sigma_z

    def F_a(self, model, q, pi):
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(self.J(model))
        F_a = scipy.sparse.diags(pi)
        if q == 'cur':
            F_a = self.L_a(model) @ F_a
        return F_a

    def F_z(self, model, q, xi):
        if numpy.isscalar(xi):
            xi = xi * numpy.ones(self.K(model))
        F_z = scipy.sparse.diags(xi)
        if q == 'cur':
            F_z = self.L_z(model) @ F_z
        return F_z

    def F_XW(self, model, q, pi):
        return self.F_a(model, q, pi)

    def F_yy(self, model, q, pi, xi):
        return (
            scipy.sparse.kron(self.F_a(model, q, pi),
                              self.H_z(model, q))
            + scipy.sparse.kron(self.H_a(model, q),
                                self.F_z(model, q, xi))
        )

    def F_Xy(self, model, q, xi):
        return scipy.sparse.kron(self.H_a(model, q),
                                 self.sigma_z(model, xi))

    def F_yw(self, model, q, xi):
        return scipy.sparse.kron(self.H_a(model, q),
                                 self.zeta_z(model) @ self.sigma_z(model, xi))

    @functools.cache
    def _get_rate_a(self, model, which):
        waiting_time = getattr(model.parameters, which)
        rate = waiting_time.rate(model.a)
        return solver.numerical.rate_make_finite(rate)

    @functools.cache
    def _get_rate_z(self, model, which):
        waiting_time = getattr(model.parameters, which)
        rate = waiting_time.rate(model.z)
        return solver.numerical.rate_make_finite(rate)

    def F(self, model, q):
        F_XW = functools.partial(self.F_XW, model, q)
        F_yy = functools.partial(self.F_yy, model, q)
        F_Xy = functools.partial(self.F_Xy, model, q)
        F_yw = functools.partial(self.F_yw, model, q)
        mu = self._get_rate_a(model, 'death')
        omega = self._get_rate_a(model, 'waning')
        rho = self._get_rate_z(model, 'progression')
        gamma = self._get_rate_z(model, 'recovery')
        return scipy.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_yy(- mu, - rho), None, None],
            [None, None, F_yw(rho), F_yy(- mu, - gamma), None],
            [None, None, None, F_Xy(gamma), F_XW(- mu)]
        ])

    @functools.cache
    def tau_a(self, model):
        nu = model.parameters.birth.maternity(model.a)
        return (model.a_step * nu).reshape((1, -1))

    def B(self, model):
        B_XW = (self.zeta_a(model)
                @ self.tau_a(model))
        B_Xy = (self.zeta_a(model)
                @ scipy.sparse.kron(self.tau_a(model),
                                    self.iota_z(model)))
        J = self.J(model)
        K = self.K(model)
        Zeros_yw = solver.Zeros((J * K, J * K))
        Zeros_yX = solver.Zeros((J * K, J))
        Zeros_Xy = solver.Zeros((J, J * K))
        Zeros_XW = solver.Zeros((J, J))
        return scipy.sparse.bmat([
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, B_XW],
            [B_XW,     B_XW,     B_Xy,     B_Xy,     Zeros_XW],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW]
        ])

    def beta(self, model):
        iota_y = scipy.sparse.kron(self.iota_a(model),
                                   self.iota_z(model))
        J = self.J(model)
        zeros_X = solver.Zeros((1, J))
        zeros_y = solver.Zeros((1, J * self.K(model)))
        return (
            model.parameters.transmission.rate
            * scipy.sparse.hstack(
                [zeros_X, zeros_X, zeros_y, iota_y, zeros_X]
            )
        )

    def T(self, model, q):
        T_a = self.H_a(model, q)
        T_XX = T_a
        T_yX = scipy.sparse.kron(T_a,
                                 self.zeta_z(model))
        J = self.J(model)
        K = self.K(model)
        Zeros_yw = solver.Zeros((J * K, J * K))
        Zeros_yX = solver.Zeros((J * K, J))
        Zeros_Xy = solver.Zeros((J, J * K))
        Zeros_XW = solver.Zeros((J, J))
        return scipy.sparse.bmat([
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW],
            [Zeros_XW, - T_XX,   Zeros_Xy, Zeros_Xy, Zeros_XW],
            [Zeros_yX, T_yX,     Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_yX, Zeros_yX, Zeros_yw, Zeros_yw, Zeros_yX],
            [Zeros_XW, Zeros_XW, Zeros_Xy, Zeros_Xy, Zeros_XW]
        ])
