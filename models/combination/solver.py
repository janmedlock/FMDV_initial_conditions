'''Solver.'''

import functools

from .. import age_structured, time_since_entry_structured, _model, _utility


class Solver(time_since_entry_structured.solver.Mixin,
             age_structured.solver.Mixin,
             _model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _jacobian_method_default = 'dense'

    @functools.cached_property
    def _zeros_az(self):
        return _utility.sparse.kron(self._zeros_a,
                                    self._zeros_z)

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        blocks = [
            self._zeros_az
            if state in self.model.states_with_z
            else self._zeros_a
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

    @functools.cached_property
    def _Zeros_a_az(self):  # pylint: disable=invalid-name
        '''Zero matrix used in constructing the other matrices.'''
        return self._zeros_a.T @ self._zeros_az

    @functools.cached_property
    def _Zeros_az_a(self):  # pylint: disable=invalid-name
        '''Zero matrix used in constructing the other matrices.'''
        return self._zeros_az.T @ self._zeros_a

    @functools.cached_property
    def _Zeros_az_az(self):  # pylint: disable=invalid-name
        '''Zero matrix used in constructing the other matrices.'''
        return self._zeros_az.T @ self._zeros_az

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        # pylint: disable-next=invalid-name
        I_az = _utility.sparse.kron(self._I_a,
                                    self._I_z)
        return _utility.sparse.block_diag([
            I_az
            if state in self.model.states_with_z
            else self._I_a
            for state in self.model.states
        ])

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        H_a = self._H_a(q)  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        H_az = _utility.sparse.kron(self._H_a(q),
                                    self._H_z(q))
        return _utility.sparse.block_diag([
            H_az
            if state in self.model.states_with_z
            else H_a
            for state in self.model.states
        ])

    def _F_XW(self, q, pi):  # pylint: disable=invalid-name
        '''A block between X states of F(q).'''
        return self._F_a(q, pi)

    def _F_az(self, q, pi, xi):  # pylint: disable=invalid-name
        '''A diagonal block for a y state of F(q).'''
        return (
            _utility.sparse.kron(self._F_a(q, pi),
                                 self._H_z(q))
            + _utility.sparse.kron(self._H_a(q),
                                   self._F_z(q, xi))
        )

    def _Sigma_a(self, q, xi):  # pylint: disable=invalid-name
        '''A block to an X state from a y state of F(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._sigma_z(xi))

    def _Sigma_az(self, q, xi):  # pylint: disable=invalid-name
        '''An off-diagonal block between y states of F(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._Sigma_z(xi))

    def F(self, q):  # pylint: disable=invalid-name
        '''Build the transition matrix F(q).'''
        mu = self._get_rate_a('death')
        omega = self._get_rate_a('waning')
        rho = self._get_rate_z('progression')
        gamma = self._get_rate_z('recovery')
        F_a = functools.partial(self._F_a, q)  # pylint: disable=invalid-name
        F_az = functools.partial(self._F_az, q)  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        Sigma_a = functools.partial(self._Sigma_a, q)
        # pylint: disable-next=invalid-name
        Sigma_az = functools.partial(self._Sigma_az, q)
        return _utility.sparse.bmat([
            [F_a(- omega - mu), None, None, None, None],
            [F_a(omega), F_a(- mu), None, None, None],
            [None, None, F_az(- mu, - rho), None, None],
            [None, None, Sigma_az(rho), F_az(- mu, - gamma), None],
            [None, None, None, Sigma_a(gamma), F_a(- mu)]
        ])

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        B_a = self._B_a  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        B_az = _utility.sparse.kron(self._B_a,
                                    self._iota_z)
        Zeros_a_a = self._Zeros_a_a  # pylint: disable=invalid-name
        Zeros_az_a = self._Zeros_az_a  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [None,       None, None, None, B_a],
            [B_a,        B_a,  B_az, B_az, None],
            [Zeros_az_a, None, None, None, None],
            [Zeros_az_a, None, None, None, None],
            [Zeros_a_a,  None, None, None, None]
        ])

    def _T_yX(self, q):  # pylint: disable=invalid-name
        '''A block to a y state from an X state of T(q).'''
        return _utility.sparse.kron(self._H_a(q),
                                    self._zeta_z)

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        H_a = self._H_a(q)  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        T_az = _utility.sparse.kron(H_a,
                                    self._zeta_z)
        Zeros_a_a = self._Zeros_a_a  # pylint: disable=invalid-name
        Zeros_a_az = self._Zeros_a_az  # pylint: disable=invalid-name
        Zeros_az_az = self._Zeros_az_az  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [Zeros_a_a, None,  Zeros_a_az, None,        None],
            [None,      - H_a, None,       None,        None],
            [None,      T_az,  None,       None,        None],
            [None,      None,  None,       Zeros_az_az, None],
            [None,      None,  None,       None,        Zeros_a_a]
        ])

    def _check_matrices(self, is_M_matrix=False):
        '''Check the solver matrices. Checking the M matrix requires
        finding the dominant eigenvalue, which is very slow and very
        memory intensive, so it is disabled by default.'''
        super()._check_matrices(is_M_matrix=is_M_matrix)
