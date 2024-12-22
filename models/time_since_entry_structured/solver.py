'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Mixin:
    '''Mixin for solvers of time-since-entry-structured models.'''

    sparse = True

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
    def _zeros_z(self):
        return _utility.sparse.Array((1, len(self.z)))

    @functools.cached_property
    def _iota_z(self):
        '''The block for integrating over time since infection.'''
        return self._integration_vector(len(self.z), self.z_step)

    @functools.cached_property
    def _zeta_z(self):
        '''The block for influx for time since infection.'''
        return self._influx_vector(len(self.z), self.z_step)

    def _sigma_z(self, xi):
        '''The vector to integrate a vector against `xi` over time
        since entry.'''
        return self._integration_against_vector(
            len(self.z), self.z_step, xi
        )

    @functools.cached_property
    def _Zeros_z_z(self):  # pylint: disable=invalid-name
        '''Zero matrix used in constructing the other matrices.'''
        return self._zeros_z.T @ self._zeros_z

    @functools.cached_property
    def _I_z(self):  # pylint: disable=invalid-name
        '''The identity matrix block for time since entry.'''
        return _utility.sparse.identity(len(self.z))

    @functools.cached_property
    def _L_z(self):  # pylint: disable=invalid-name
        '''The lag matrix in time since entry.'''
        return self._lag_matrix(len(self.z))

    def _H_z(self, q):  # pylint: disable=invalid-name
        '''The time-since-entry diagonal block of H(q).'''
        if q == 'new':
            H_z = self._I_z  # pylint: disable=invalid-name
        elif q == 'cur':
            H_z = self._L_z  # pylint: disable=invalid-name
        else:
            raise ValueError(f'{q=}!')
        return H_z

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

    def _get_rate_z(self, which):
        '''Get the time-since-entry-dependent rate `which` and make
        finite any infinite entries.'''
        waiting_time = getattr(self.model.parameters, which)
        rate = waiting_time.rate(self.z)
        return _utility.numerical.rate_make_finite(rate)


class Solver(Mixin, _model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _jacobian_method_default = 'sparse_csc'

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        blocks = [
            self._zeros_z
            if state in self.model.states_with_z
            else 0
            for state in self.model.states
        ]
        infectious = self.model.states.index('infectious')
        assert 'infectious' in self.model.states_with_z
        blocks[infectious] = self._iota_z
        return (
            self.model.parameters.transmission.rate
            * _utility.sparse.hstack(blocks)
        )

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return _utility.sparse.block_diag([
            self._I_z
            if state in self.model.states_with_z
            else 1
            for state in self.model.states
        ])

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        H_z = self._H_z(q)  # pylint: disable=invalid-name
        return _utility.sparse.block_diag([
            H_z
            if state in self.model.states_with_z
            else 1
            for state in self.model.states
        ])

    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''
        mu = self.model.parameters.death_rate_mean
        omega = 1 / self.model.parameters.waning.mean
        rho = self._get_rate_z('progression')
        gamma = self._get_rate_z('recovery')
        F_z = functools.partial(self._F_z, q)  # pylint: disable=invalid-name
        sigma_z = self._sigma_z
        zeta_z = self._zeta_z
        return _utility.sparse.bmat([
            [- omega - mu, None, None, None, None],
            [omega, - mu, None, None, None],
            [None, None, F_z(- rho - mu), None, None],
            [None, None, zeta_z @ sigma_z(rho), F_z(- gamma - mu), None],
            [None, None, None, sigma_z(gamma), - mu]
        ])

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        iota_z = self._iota_z
        zeros_z_T = self._zeros_z.T  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [None,      None, None,   None,   1],
            [1,         1,    iota_z, iota_z, None],
            [zeros_z_T, None, None,   None,   None],
            [zeros_z_T, None, None,   None,   None],
            [0,         None, None,   None,   None]
        ])

    @functools.cached_property
    def _T_(self):  # pylint: disable=invalid-name
        '''T is independent of q, so build once and reuse it.'''
        zeta_z = self._zeta_z
        zeros_z = self._zeros_z
        Zeros_z_z = self._Zeros_z_z  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [0,    None,   zeros_z, None,      None],
            [None, - 1,    None,    None,      None],
            [None, zeta_z, None,    None,      None],
            [None, None,   None,    Zeros_z_z, None],
            [None, None,   None,    None,      0]
        ])

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        return self._T_
