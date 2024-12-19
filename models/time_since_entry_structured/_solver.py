'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    sparse = True

    _jacobian_method_default = 'sparse_csc'

    @property
    def z_max(self):
        '''The maximum time since entry.'''
        return self.model.z_max

    @property
    def z_step(self):
        '''The step size in time since entry.'''
        return self.t_step

    @functools.cached_property
    def z(self):
        '''The time-since-entry vector.'''
        return _utility.numerical.build_t(0, self.z_max, self.z_step)

    @functools.cached_property
    def _iota(self):
        '''The block for integrating over time since infection.'''
        return self._integration_vector(len(self.z), self.z_step)

    @functools.cached_property
    def _zeta(self):
        '''The block for influx for time since infection.'''
        return self._influx_vector(len(self.z), self.z_step)

    @functools.cached_property
    def Zeros(self):  # pylint: disable=invalid-name
        '''Zero matrices used in constructing the other matrices.'''
        len_X = 1  # pylint: disable=invalid-name
        len_y = len(self.z)
        return {
            'XW': _utility.sparse.Array((len_X, len_X)),
            'Xy': _utility.sparse.Array((len_X, len_y)),
            'yw': _utility.sparse.Array((len_y, len_y)),
        }

    @functools.cached_property
    def _I_XX(self):  # pylint: disable=invalid-name
        '''An identity matrix block for an X state.'''
        return _utility.sparse.identity(1)

    @functools.cached_property
    def _I_yy(self):  # pylint: disable=invalid-name
        '''An identity matrix block for a y state.'''
        return _utility.sparse.identity(len(self.z))

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return _utility.sparse.block_diag([
            self._I_yy if state in self.model.states_with_z else self._I_XX
            for state in self.model.states
        ])

    @functools.cached_property
    def _L_yy(self):  # pylint: disable=invalid-name
        '''The lag matrix for a y state.'''
        return self._lag_matrix(len(self.z))

    @property
    def _h_XX(self):  # pylint: disable=invalid-name
        '''A diagonal block for an X state of H(q).'''
        return self._I_XX

    def _H_yy(self, q):  # pylint: disable=invalid-name
        '''A diagonal block for a y state of H(q).'''
        if q == 'new':
            H_yy = self._I_yy  # pylint: disable=invalid-name
        elif q == 'cur':
            H_yy = self._L_yy  # pylint: disable=invalid-name
        else:
            raise ValueError(f'{q=}!')
        return H_yy

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        H_yy = self._H_yy(q)  # pylint: disable=invalid-name
        return _utility.sparse.block_diag([
            H_yy if state in self.model.states_with_z else self._h_XX
            for state in self.model.states
        ])

    def _sigma(self, xi):
        '''The vector to integrate a y state against `xi`.'''
        return self._integration_against_vector(
            len(self.z), self.z_step, xi
        )

    def _f_XX(self, pi):  # pylint: disable=invalid-name
        '''A diagonal block for an X state of F(q).'''
        return _utility.sparse.Array([[pi]])

    def _F_yy(self, q, xi):  # pylint: disable=invalid-name
        '''A diagonal block for a y state of F(q).'''
        if q not in self._q_vals:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(xi):
            xi *= numpy.ones(len(self.z))
        F_yy = _utility.sparse.diags(xi)  # pylint: disable=invalid-name
        if q == 'cur':
            F_yy = self._L_yy @ F_yy  # pylint: disable=invalid-name
        return F_yy

    def _f_Xy(self, xi):  # pylint: disable=invalid-name
        '''A block to an X state from a y state of F(q).'''
        return self._sigma(xi)

    def _F_yw(self, xi):  # pylint: disable=invalid-name
        '''An off-diagonal block between y states of F(q).'''
        return self._zeta @ self._sigma(xi)

    def _get_rate(self, which):
        '''Get the rate `which` and make finite any infinite entries.'''
        waiting_time = getattr(self.model.parameters, which)
        rate = waiting_time.rate(self.z)
        return _utility.numerical.rate_make_finite(rate)

    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''
        mu = self.model.parameters.death_rate_mean
        omega = self._get_rate('waning')
        rho = self._get_rate('progression')
        gamma = self._get_rate('recovery')
        f_XX = self._f_XX  # pylint: disable=invalid-name
        f_Xy = self._f_Xy  # pylint: disable=invalid-name
        F_yy = functools.partial(self._F_yy, q)  # pylint: disable=invalid-name
        F_yw = self._F_yw  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [F_yy(- omega - mu), None, None, None, None],
            [f_Xy(omega), f_XX(- mu), None, None, None],
            [None, None, F_yy(- rho - mu), None, None],
            [None, None, F_yw(rho), F_yy(- gamma - mu), None],
            [None, None, None, f_Xy(gamma), f_XX(- mu)]
        ])

    @property
    def _b_XX(self):  # pylint: disable=invalid-name
        '''A diagonal block for an X state of B.'''
        return self._I_XX

    @property
    def _b_Xy(self):  # pylint: disable=invalid-name
        '''A block to an X state from a y state of B.'''
        return self._iota

    @property
    def _b_yX(self):  # pylint: disable=invalid-name
        '''A block to a y state from an X state of B.'''
        return self._zeta

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        b_XX = self._b_XX  # pylint: disable=invalid-name
        b_yX = self._b_yX  # pylint: disable=invalid-name
        b_Xy = self._b_Xy  # pylint: disable=invalid-name
        zeros_Xy = self.Zeros['Xy']  # pylint: disable=invalid-name
        Zeros_yw = self.Zeros['yw']  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [None,     None, None, None, b_yX],
            [b_Xy,     b_XX, b_Xy, b_Xy, None],
            [Zeros_yw, None, None, None, None],
            [Zeros_yw, None, None, None, None],
            [zeros_Xy, None, None, None, None]
        ])

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        zeros_X = self.Zeros['XW']  # pylint: disable=invalid-name
        zeros_y = self.Zeros['Xy']
        blocks = [
            zeros_y if state in self.model.states_with_z else zeros_X
            for state in self.model.states
        ]
        infectious = self.model.states.index('infectious')
        assert 'infectious' in self.model.states_with_z
        blocks[infectious] = self._iota
        return (
            self.model.parameters.transmission.rate
            * _utility.sparse.hstack(blocks)
        )

    @property
    def _t_XX(self):  # pylint: disable=invalid-name
        '''A diagonal block for an X state of T(q).'''
        return self._h_XX

    @property
    def _t_yX(self):  # pylint: disable=invalid-name
        '''A block to a y state from an X state of T(q).'''
        return self._zeta

    @functools.cached_property
    def _T_(self):  # pylint: disable=invalid-name
        '''T is independent of q, so build once and reuse it.'''
        t_XX = self._t_XX  # pylint: disable=invalid-name
        t_yX = self._t_yX  # pylint: disable=invalid-name
        zeros_XW = self.Zeros['XW']  # pylint: disable=invalid-name
        Zeros_yw = self.Zeros['yw']  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [Zeros_yw, None,   Zeros_yw, None,     None],
            [None,     - t_XX, None,     None,     None],
            [None,     t_yX,   None,     None,     None],
            [None,     None,   None,     Zeros_yw, None],
            [None,     None,   None,     None,     zeros_XW]
        ])

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        return self._T_
