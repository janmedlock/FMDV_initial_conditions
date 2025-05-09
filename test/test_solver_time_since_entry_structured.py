#!/usr/bin/python3
'''Test the solver matrices for the time-since-entry-structured model.'''

import functools

import numpy
import scipy.sparse

from context import models

import solver


class TestSolver(solver.TestSolver):
    '''Test the time-since-entry-structured solver.'''

    Model = models.time_since_entry_structured.Model

    @functools.cache
    def K(self, model):
        return len(model.z)

    @functools.cache
    def iota(self, model):
        return (model.z_step * numpy.ones((1, self.K(model))))

    @functools.cache
    def zeta(self, model):
        zeta = scipy.sparse.dok_array((self.K(model), 1))
        zeta[0, 0] = 1 / model.z_step
        return zeta

    @functools.cache
    def L_yy(self, model):
        K = self.K(model)
        return solver.sparse.diags_from_dict({
            -1: numpy.ones(K - 1),
            0: numpy.hstack([numpy.zeros(K - 1), 1]),
        })

    @functools.cache
    def H(self, model, q):
        H_yy = scipy.sparse.identity(self.K(model))
        if q == 'cur':
            H_yy = self.L_yy(model) @ H_yy
        H_XX = [[1]]
        return scipy.sparse.block_diag([H_XX, H_XX, H_yy, H_yy, H_XX])

    def sigma(self, model, xi):
        K = self.K(model)
        if numpy.isscalar(xi):
            xi = xi * numpy.ones(K)
        sigma = scipy.sparse.dok_array((1, K))
        sigma[0] = model.z_step * xi
        return sigma

    @functools.cache
    def f_XW(self, pi):
        return [[pi]]

    def F_yy(self, model, q, xi):
        if numpy.isscalar(xi):
            xi = xi * numpy.ones(self.K(model))
        F_yy = scipy.sparse.diags(xi)
        if q == 'cur':
            F_yy = self.L_yy(model) @ F_yy
        return F_yy

    def F_yw(self, model, xi):
        return self.zeta(model) @ self.sigma(model, xi)

    def f_Xy(self, model, xi):
        return self.sigma(model, xi)

    @functools.cache
    def _get_rate(self, model, which):
        waiting_time = getattr(model.parameters, which)
        rate = waiting_time.rate(model.z)
        return solver.numerical.rate_make_finite(rate)

    def F(self, model, q):
        f_XW = self.f_XW
        F_yy = functools.partial(self.F_yy, model, q)
        F_yw = functools.partial(self.F_yw, model)
        f_Xy = functools.partial(self.f_Xy, model)
        mu = model.parameters.death_rate_mean
        omega = 1 / model.parameters.waning.mean
        rho = self._get_rate(model, 'progression')
        gamma = self._get_rate(model, 'recovery')
        return scipy.sparse.bmat([
            [f_XW(- omega - mu), None, None, None, None],
            [f_XW(omega), f_XW(- mu), None, None, None],
            [None, None, F_yy(- rho - mu), None, None],
            [None, None, F_yw(rho), F_yy(- gamma - mu), None],
            [None, None, None, f_Xy(gamma), f_XW(- mu)]
        ])

    def B(self, model):
        K = self.K(model)
        b_Xy = self.iota(model)
        b_XX = [[1]]
        Zeros_yw = solver.Zeros((K, K))
        zeros_yX = solver.Zeros((K, 1))
        zeros_Xy = solver.Zeros((1, K))
        zeros_XW = solver.Zeros((1, 1))
        return scipy.sparse.bmat([
            [zeros_XW, zeros_XW, zeros_Xy, zeros_Xy, b_XX],
            [b_XX, b_XX, b_Xy, b_Xy, zeros_XW],
            [zeros_yX, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_yX, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_XW, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW]
        ])

    def beta(self, model):
        iota = self.iota(model)
        zeros_X = solver.Zeros((1, 1))
        zeros_y = solver.Zeros((1, self.K(model)))
        return (
            model.parameters.transmission.rate
            * scipy.sparse.hstack(
                [zeros_X, zeros_X, zeros_y, iota, zeros_X]
            )
        )

    @functools.cache
    def _T(self, model):
        # `T` is independent of `q`.
        K = self.K(model)
        t_XX = numpy.array([[1]])
        t_yX = self.zeta(model)
        Zeros_yw = solver.Zeros((K, K))
        zeros_yX = solver.Zeros((K, 1))
        zeros_Xy = solver.Zeros((1, K))
        zeros_XW = solver.Zeros((1, 1))
        return scipy.sparse.bmat([
            [zeros_XW, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW],
            [zeros_XW, - t_XX, zeros_Xy, zeros_Xy, zeros_XW],
            [zeros_yX, t_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_yX, zeros_yX, Zeros_yw, Zeros_yw, zeros_yX],
            [zeros_XW, zeros_XW, zeros_Xy, zeros_Xy, zeros_XW]
        ])

    def T(self, model, q):
        return self._T(model)
