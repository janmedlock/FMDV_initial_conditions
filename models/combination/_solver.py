'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    sparse = True

    _jacobian_method_default = 'dense'

    @property
    def a_max(self):
        '''The maximum age.'''
        return self.parameters.age_max

    @property
    def a_step(self):
        '''The step size in age.'''
        return self.t_step

    @functools.cached_property
    def a(self):
        '''The solution ages.'''
        return _utility.numerical.build_t(0, self.a_max, self.a_step)

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
    def _iota_a(self):
        '''The block for integrating over age.'''
        return self._integration_vector(len(self.a), self.a_step)

    @functools.cached_property
    def _iota_z(self):
        '''The block for integrating over time since entry.'''
        return self._integration_vector(len(self.z), self.z_step)

    @functools.cached_property
    def _zeta_a(self):
        '''The influx block for age.'''
        return self._influx_vector(len(self.a), self.a_step)

    @functools.cached_property
    def _zeta_z(self):
        '''The influx block for time since entry.'''
        return self._influx_vector(len(self.z), self.z_step)

    @functools.cached_property
    def Zeros(self):  # pylint: disable=invalid-name
        '''Zero matrices used in constructing the other matrices.'''
        len_X = len(self.a)  # pylint: disable=invalid-name
        len_y = len(self.a) * len(self.z)
        return {
            'XW': _utility.sparse.Array((len_X, len_X)),
            'Xy': _utility.sparse.Array((len_X, len_y)),
            'yX': _utility.sparse.Array((len_y, len_X)),
            'yw': _utility.sparse.Array((len_y, len_y)),
        }

    @functools.cached_property
    def _I_a(self):  # pylint: disable=invalid-name
        '''The identity matrix block for age.'''
        return _utility.sparse.identity(len(self.a))

    @functools.cached_property
    def _I_z(self):  # pylint: disable=invalid-name
        '''The identity matrix block for time since entry.'''
        return _utility.sparse.identity(len(self.z))

    @functools.cached_property
    def _I_XX(self):  # pylint: disable=invalid-name
        '''The identity matrix block for an X state.'''
        return self._I_a

    @functools.cached_property
    def _I_yy(self):  # pylint: disable=invalid-name
        '''The identity matrix block for a y state.'''
        return _utility.sparse.kron(self._I_a, self._I_z)

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return _utility.sparse.block_diag([
            self._I_yy if state in self.model.states_with_z else self._I_XX
            for state in self.model.states
        ])

    @functools.cached_property
    def _L_a(self):  # pylint: disable=invalid-name
        '''The lag matrix in age.'''
        return self._lag_matrix(len(self.a))

    @functools.cached_property
    def _L_z(self):  # pylint: disable=invalid-name
        '''The lag matrix in time since entry.'''
        return self._lag_matrix(len(self.z))

    def _H_a(self, q):  # pylint: disable=invalid-name
        '''The age block of H(q).'''
        if q == 'new':
            H_a = self._I_a  # pylint: disable=invalid-name
        elif q == 'cur':
            H_a = self._L_a  # pylint: disable=invalid-name
        else:
            raise ValueError(f'{q=}!')
        return H_a

    def _H_z(self, q):  # pylint: disable=invalid-name
        '''The time-since-entry block of H(q).'''
        if q == 'new':
            H_z = self._I_z  # pylint: disable=invalid-name
        elif q == 'cur':
            H_z = self._L_z  # pylint: disable=invalid-name
        else:
            raise ValueError(f'{q=}!')
        return H_z

    def _H_XX(self, q):  # pylint: disable=invalid-name
        '''The diagonal block for an X state of H(q).'''
        return self._H_a(q)

    def _H_yy(self, q):  # pylint: disable=invalid-name
        '''The diagonal block for a y state of H(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._H_z(q))

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        H_XX = self._H_XX(q)  # pylint: disable=invalid-name
        H_yy = self._H_yy(q)  # pylint: disable=invalid-name
        return _utility.sparse.block_diag([
            H_yy if state in self.model.states_with_z else H_XX
            for state in self.model.states
        ])

    def _sigma_z(self, xi):
        '''The vector to integrate a vector against `xi` over time
        since entry.'''
        return self._integration_against_vector(
            len(self.z), self.z_step, xi
        )

    def _F_a(self, q, pi):  # pylint: disable=invalid-name
        '''An age block of F(q).'''
        if q not in self._q_vals:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(pi):
            pi *= numpy.ones(len(self.a))
        F_a = _utility.sparse.diags(pi)  # pylint: disable=invalid-name
        if q == 'cur':
            F_a = self._L_a @ F_a  # pylint: disable=invalid-name
        return F_a

    def _F_z(self, q, xi):  # pylint: disable=invalid-name
        '''A time-since-entry block of F(q).'''
        if q not in self._q_vals:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(xi):
            xi *= numpy.ones(len(self.z))
        F_z = _utility.sparse.diags(xi)  # pylint: disable=invalid-name
        if q == 'cur':
            F_z = self._L_z @ F_z  # pylint: disable=invalid-name
        return F_z

    def _F_XW(self, q, pi):  # pylint: disable=invalid-name
        '''A block between X states of F(q).'''
        return self._F_a(q, pi)

    def _F_yy(self, q, pi, xi):  # pylint: disable=invalid-name
        '''A diagonal block for a y state of F(q).'''
        return (
            _utility.sparse.kron(self._F_a(q, pi),
                                 self._H_z(q))
            + _utility.sparse.kron(self._H_a(q),
                                   self._F_z(q, xi))
        )

    def _F_Xy(self, q, xi):  # pylint: disable=invalid-name
        '''A block to an X state from a y state of F(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._sigma_z(xi))

    def _F_yw(self, q, xi):  # pylint: disable=invalid-name
        '''An off-diagonal block between y states of F(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._zeta_z @ self._sigma_z(xi))

    def _get_rate_a(self, which):
        '''Get the age-dependent rate `which` and make finite any
        infinite entries.'''
        waiting_time = getattr(self.model.parameters, which)
        rate = waiting_time.rate(self.a)
        return _utility.numerical.rate_make_finite(rate)

    def _get_rate_z(self, which):
        '''Get the time-since-entry-dependent rate `which` and make
        finite any infinite entries.'''
        waiting_time = getattr(self.model.parameters, which)
        rate = waiting_time.rate(self.z)
        return _utility.numerical.rate_make_finite(rate)

    def F(self, q):  # pylint: disable=invalid-name
        '''Build the transition matrix F(q).'''
        mu = self._get_rate_a('death')
        omega = self._get_rate_a('waning')
        rho = self._get_rate_z('progression')
        gamma = self._get_rate_z('recovery')
        F_XW = functools.partial(self._F_XW, q)  # pylint: disable=invalid-name
        F_yy = functools.partial(self._F_yy, q)  # pylint: disable=invalid-name
        F_Xy = functools.partial(self._F_Xy, q)  # pylint: disable=invalid-name
        F_yw = functools.partial(self._F_yw, q)  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_yy(- mu, - rho), None, None],
            [None, None, F_yw(rho), F_yy(- mu, - gamma), None],
            [None, None, None, F_Xy(gamma), F_XW(- mu)]
        ])

    @functools.cached_property
    def _tau_a(self):
        '''The maternity integration vector for age.'''
        nu = self.model.parameters.birth.maternity(self.a)
        tau_a = _utility.sparse.Array(self.a_step * nu)
        return tau_a

    @property
    def _B_XW(self):  # pylint: disable=invalid-name
        '''A block between X states of B.'''
        return self._zeta_a @ self._tau_a

    @property
    def _B_Xy(self):  # pylint: disable=invalid-name
        '''A block to an X state from a y state of B.'''
        return (
            self._zeta_a
            @ _utility.sparse.kron(self._tau_a,
                                   self._iota_z)
        )

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        B_XW = self._B_XW  # pylint: disable=invalid-name
        B_Xy = self._B_Xy  # pylint: disable=invalid-name
        Zeros_XW = self.Zeros['XW']  # pylint: disable=invalid-name
        Zeros_yX = self.Zeros['yX']  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [None,     None, None, None, B_XW],
            [B_XW,     B_XW, B_Xy, B_Xy, None],
            [Zeros_yX, None, None, None, None],
            [Zeros_yX, None, None, None, None],
            [Zeros_XW, None, None, None, None]
        ])

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        zeros_a = _utility.sparse.Array((1, len(self.a)))
        zeros_z = _utility.sparse.Array((1, len(self.z)))
        zeros_X = zeros_a  # pylint: disable=invalid-name
        zeros_y = _utility.sparse.kron(zeros_a,
                                       zeros_z)
        blocks = [
            zeros_y if state in self.model.states_with_z else zeros_X
            for state in self.model.states
        ]
        infectious = self.model.states.index('infectious')
        assert 'infectious' in self.model.states_with_z
        blocks[infectious] = _utility.sparse.kron(self._iota_a,
                                                  self._iota_z)
        return (
            self.model.parameters.transmission.rate
            * _utility.sparse.hstack(blocks)
        )

    def _T_XX(self, q):  # pylint: disable=invalid-name
        '''A diagonal block for an X state of T(q).'''
        return self._H_a(q)

    def _T_yX(self, q):  # pylint: disable=invalid-name
        '''A block to a y state from an X state of T(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._zeta_z)

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        T_XX = self._T_XX(q)  # pylint: disable=invalid-name
        T_yX = self._T_yX(q)  # pylint: disable=invalid-name
        Zeros_XW = self.Zeros['XW']  # pylint: disable=invalid-name
        Zeros_Xy = self.Zeros['Xy']  # pylint: disable=invalid-name
        Zeros_yw = self.Zeros['yw']  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [Zeros_XW, None,   Zeros_Xy, None,     None],
            [None,     - T_XX, None,     None,     None],
            [None,     T_yX,   None,     None,     None],
            [None,     None,   None,     Zeros_yw, None],
            [None,     None,   None,     None,     Zeros_XW]
        ])

    def _check_matrices(self, is_M_matrix=False):
        '''Check the solver matrices. Checking the M matrix requires
        finding the dominant eigenvalue, which is very slow and very
        memory intensive, so it is disabled by default.'''
        super()._check_matrices(is_M_matrix=is_M_matrix)
