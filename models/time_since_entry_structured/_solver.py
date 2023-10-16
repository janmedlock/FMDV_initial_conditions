'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _sparse = True

    _jacobian_method_default = 'sparse_csc'

    def __init__(self, model, **kwds):
        self.z = model.z
        super().__init__(model, **kwds)

    @staticmethod
    def _get_z_step(t_step):
        '''Get the step size in time since entry.'''
        z_step = t_step
        assert z_step > 0
        return z_step

    @functools.cached_property
    def z_step(self):
        '''The step size in time since entry.'''
        z_step = self._get_z_step(self.t_step)
        return z_step

    @functools.cached_property
    def Zeros(self):
        '''Zero matrices used in constructing the other matrices.'''
        K = len(self.z)
        Zeros = {
            'XW': _utility.sparse.Array((1, 1)),
            'Xy': _utility.sparse.Array((1, K)),
            'yw': _utility.sparse.Array((K, K))
        }
        return Zeros

    @functools.cached_property
    def I_XX(self):
        '''Build an identity matrix block for an X state.'''
        I_XX = _utility.sparse.identity(1)
        return I_XX

    @functools.cached_property
    def I_yy(self):
        '''Build an identity matrix block for a y state.'''
        K = len(self.z)
        I_yy = _utility.sparse.identity(K)
        return I_yy

    @functools.cached_property
    def L_yy(self):
        '''Build the lag matrix for a y state.'''
        K = len(self.z)
        diags = {
            -1: numpy.ones(K - 1),
            0: numpy.hstack([numpy.zeros(K - 1), 1])
        }
        L_yy = _utility.sparse.diags_from_dict(diags)
        return L_yy

    @functools.cached_property
    def zeta(self):
        '''Build the vector for entering a y state.'''
        K = len(self.z)
        zeta = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.z_step},
            shape=(K, 1)
        )
        return zeta

    def _iota_y(self):
        '''Build a block for integrating a y state over time since
        infection.'''
        K = len(self.z)
        iota_y = self.z_step * numpy.ones((1, K))
        return iota_y

    def _sigma(self, xi):
        '''Build the vector to integrate a y state over `xi`.'''
        if numpy.isscalar(xi):
            K = len(self.z)
            xi *= numpy.ones(K)
        sigma = _utility.sparse.Array(self.z_step * xi)
        return sigma

    def _I(self):
        '''Build the identity matrix.'''
        I_XX = self.I_XX
        I_yy = self.I_yy
        blocks = [
            I_yy if state in self.model.states_with_z else I_XX
            for state in self.model.states
        ]
        I = _utility.sparse.block_diag(blocks)
        return I

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        zeros_X = self.Zeros['XW']
        zeros_y = self.Zeros['Xy']
        blocks = [
            zeros_y if state in self.model.states_with_z else zeros_X
            for state in self.model.states
        ]
        infectious = self.model.states.index('infectious')
        assert 'infectious' in self.model.states_with_z
        blocks[infectious] = self._iota_y()
        beta = (self.model.parameters.transmission.rate
                * _utility.sparse.hstack(blocks))
        return beta

    def _h_XX(self):
        '''Build a diagonal block for an X state of H(q).'''
        h_XX = self.I_XX
        return h_XX

    def _H_yy(self, q):
        '''Build a diagonal block for a y state of H(q).'''
        if q == 'new':
            H_yy = self.I_yy
        elif q == 'cur':
            H_yy = self.L_yy
        else:
            raise ValueError(f'{q=}!')
        return H_yy

    def _H(self, q):
        '''Build the time-step matrix H(q).'''
        h_XX = self._h_XX()
        H_yy = self._H_yy(q)
        blocks = [
            H_yy if state in self.model.states_with_z else h_XX
            for state in self.model.states
        ]
        H = _utility.sparse.block_diag(blocks)
        return H

    def _f_XX(self, pi):
        '''Build a diagonal block for an X state of F(q).'''
        f_XX = _utility.sparse.Array([[pi]])
        return f_XX

    def _F_yy(self, q, xi):
        '''Build a diagonal block for a y state of F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(xi):
            K = len(self.z)
            xi *= numpy.ones(K)
        F_yy = _utility.sparse.diags(xi)
        if q == 'cur':
            F_yy = self.L_yy @ F_yy
        return F_yy

    def _f_Xy(self, xi):
        '''Build a block to an X state from a y state of F(q).'''
        f_Xy = self._sigma(xi)
        return f_Xy

    def _F_yw(self, xi):
        '''Build an off-diagonal block between y states of F(q).'''
        F_yw = self.zeta @ self._sigma(xi)
        return F_yw

    def _get_rate(self, which):
        '''Get the rate `which` and make finite any infinite entries.'''
        param = getattr(self.model.parameters, which)
        rate = param.rate(self.z)
        return _utility.numerical.rate_make_finite(rate)

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        mu = self.model.parameters.death_rate_mean
        omega = self._get_rate('waning')
        rho = self._get_rate('progression')
        gamma = self._get_rate('recovery')
        f_XX = self._f_XX
        f_Xy = self._f_Xy
        F_yy = functools.partial(self._F_yy, q)
        F_yw = self._F_yw
        F = _utility.sparse.bmat([
            [F_yy(- omega - mu), None, None, None, None],
            [f_Xy(omega), f_XX(- mu), None, None, None],
            [None, None, F_yy(- rho - mu), None, None],
            [None, None, F_yw(rho), F_yy(- gamma - mu), None],
            [None, None, None, f_Xy(gamma), f_XX(- mu)]
        ])
        return F

    def _t_XX(self):
        '''Build a diagonal block for an X state of T(q).'''
        t_XX = self.I_XX
        return t_XX

    def _t_yX(self):
        '''Build a block to a y state from an X state of T(q).'''
        t_yX = self.zeta
        return t_yX

    @functools.cached_property
    def _T_(self):
        '''T is independent of q, so built once and reuse it.'''
        t_XX = self._t_XX()
        t_yX = self._t_yX()
        zeros_XW = self.Zeros['XW']
        Zeros_yw = self.Zeros['yw']
        T = _utility.sparse.bmat([
            [Zeros_yw,   None, Zeros_yw,     None,     None],
            [     None, - t_XX,    None,     None,     None],
            [     None,   t_yX,    None,     None,     None],
            [     None,   None,    None, Zeros_yw,     None],
            [     None,   None,    None,     None, zeros_XW]
        ])
        return T

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        T = self._T_
        return T

    def _b_XX(self):
        '''Build a diagonal block for an X state of B.'''
        b_XX = self.I_XX
        return b_XX

    def _b_Xy(self):
        '''Build a block to an X state from a y state of B.'''
        K = len(self.z)
        tau = _utility.sparse.Array(self.z_step * numpy.ones(shape=(1, K)))
        b_Xy = tau
        return b_Xy

    def _b_yX(self):
        '''Build a block to a y state from an X state of B.'''
        b_yX = self.zeta
        return b_yX

    def _B(self):
        '''Build the birth matrix B.'''
        b_XX = self._b_XX()
        b_yX = self._b_yX()
        b_Xy = self._b_Xy()
        zeros_Xy = self.Zeros['Xy']
        Zeros_yw = self.Zeros['yw']
        B = _utility.sparse.bmat([
            [    None, None, None, None, b_yX],
            [    b_Xy, b_XX, b_Xy, b_Xy, None],
            [Zeros_yw, None, None, None, None],
            [Zeros_yw, None, None, None, None],
            [zeros_Xy, None, None, None, None]
        ])
        return B
