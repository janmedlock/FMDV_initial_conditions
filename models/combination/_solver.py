'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _sparse = True

    _jacobian_method_default = 'dense'

    def __init__(self, model, **kwds):
        self.a = model.a
        self.z = model.z
        super().__init__(model, **kwds)

    @staticmethod
    def _get_a_step(t_step):
        a_step = t_step
        assert a_step > 0
        return a_step

    @staticmethod
    def _get_z_step(t_step):
        z_step = t_step
        assert z_step > 0
        return z_step

    @functools.cached_property
    def a_step(self):
        return self._get_a_step(self.t_step)

    @functools.cached_property
    def z_step(self):
        return self._get_z_step(self.t_step)

    @functools.cached_property
    def Zeros(self):
        '''Zero matrices used in constructing the other matrices.'''
        J = len(self.a)
        K = len(self.z)
        Zeros = {
            'XW': _utility.sparse.array((J, J)),
            'Xy': _utility.sparse.array((J, J * K)),
            'yX': _utility.sparse.array((J * K, J)),
            'yw': _utility.sparse.array((J * K, J * K))
        }
        return Zeros

    @functools.cached_property
    def I_a(self):
        '''Build an identity matrix block for age.'''
        J = len(self.a)
        I_a = _utility.sparse.identity(J)
        return I_a

    @functools.cached_property
    def I_z(self):
        '''Build an identity matrix block for time since entry.'''
        K = len(self.z)
        I_z = _utility.sparse.identity(K)
        return I_z

    @property
    def I_XX(self):
        '''Build an identity matrix block for an X state.'''
        I_XX = self.I_a
        return I_XX

    @functools.cached_property
    def I_yy(self):
        '''Build an identity matrix block for a y state.'''
        I_yy = _utility.sparse.kron(self.I_a, self.I_z)
        return I_yy

    @staticmethod
    def _L(C):
        '''Build the lag matrix of shape CxC.'''
        diags = {
            -1: numpy.ones(C - 1),
            0: numpy.hstack([numpy.zeros(C - 1), 1])
        }
        L = _utility.sparse.diags_from_dict(diags)
        return L

    @functools.cached_property
    def L_a(self):
        '''Build the lag matrix in age.'''
        J = len(self.a)
        L_a = self._L(J)
        return L_a

    @functools.cached_property
    def L_z(self):
        '''Build the lag matrix in time since entry.'''
        K = len(self.z)
        L_z = self._L(K)
        return L_z

    @functools.cached_property
    def b_a(self):
        '''Build the vector for entering a state in age.'''
        J = len(self.a)
        b_a = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.a_step},
            shape=(J, 1)
        )
        return b_a

    @functools.cached_property
    def zeta_z(self):
        '''Build the vector for entering a state in time since entry.'''
        K = len(self.z)
        zeta_z = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.z_step},
            shape=(K, 1)
        )
        return zeta_z

    def _iota_a(self):
        '''Build a block for integrating over age.'''
        J = len(self.a)
        iota_a = self.a_step * numpy.ones((1, J))
        return iota_a

    @functools.cached_property
    def iota_z(self):
        '''Build a block for integrating over time since entry.'''
        K = len(self.z)
        iota_z = self.z_step * numpy.ones((1, K))
        return iota_z

    def _iota_y(self):
        '''Build a block for integrating a y state over age and time
        since entry.'''
        iota_a = self._iota_a()
        iota_z = self.iota_z
        iota_y = numpy.kron(iota_a, iota_z)
        return iota_y

    def _sigma_z(self, xi):
        '''Build the vector to integrate a vector times `xi` over time
        since entry.'''
        if numpy.isscalar(xi):
            K = len(self.z)
            xi *= numpy.ones(K)
        sigma = _utility.sparse.array(self.z_step * xi)
        return sigma

    @functools.cached_property
    def tau_a(self):
        '''Build the vector to integrate an vector in age over the
        age-dependent maternity.'''
        nu = self.model.parameters.birth.maternity(self.a)
        tau_a = _utility.sparse.array(self.a_step * nu)
        return tau_a

    def _tau_X(self):
        '''Build the vector to integrate an X state over the
        age-dependent maternity.'''
        tau_X = self.tau_a
        return tau_X

    def _tau_y(self):
        '''Build the vector to integrate a y state over the
        age-dependent maternity.'''
        tau_a = self.tau_a
        iota_z = self.iota_z
        tau_y = _utility.sparse.kron(tau_a, iota_z)
        return tau_y

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
        J = len(self.a)
        K = len(self.z)
        zeros_X = _utility.sparse.array((1, J))
        zeros_y = _utility.sparse.array((1, J * K))
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

    def _H_a(self, q):
        '''Build an age block of H(q).'''
        if q == 'new':
            H_a = self.I_a
        elif q == 'cur':
            H_a = self.L_a
        else:
            raise ValueError(f'{q=}!')
        return H_a

    def _H_z(self, q):
        '''Build a time-since-entry block of H(q).'''
        if q == 'new':
            H_z = self.I_z
        elif q == 'cur':
            H_z = self.L_z
        else:
            raise ValueError(f'{q=}!')
        return H_z

    def _H_XX(self, q):
        '''Build a diagonal block for an X state of H(q).'''
        H_XX = self._H_a(q)
        return H_XX

    def _H_yy(self, q):
        '''Build a diagonal block for a y state of H(q).'''
        H_a = self._H_a(q)
        H_z = self._H_z(q)
        H_yy = _utility.sparse.kron(H_a, H_z)
        return H_yy

    def _H(self, q):
        '''Build the time-step matrix H(q).'''
        H_XX = self._H_XX(q)
        H_yy = self._H_yy(q)
        blocks = [
            H_yy if state in self.model.states_with_z else H_XX
            for state in self.model.states
        ]
        H = _utility.sparse.block_diag(blocks)
        return H

    def _F_a(self, q, pi):
        '''Build an age block of F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(pi):
            J = len(self.a)
            pi *= numpy.ones(J)
        F_a = _utility.sparse.diags(pi)
        if q == 'cur':
            F_a = self.L_a @ F_a
        return F_a

    def _F_z(self, q, xi):
        '''Build a time-since-entry block of F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(xi):
            K = len(self.z)
            xi *= numpy.ones(K)
        F_z = _utility.sparse.diags(xi)
        if q == 'cur':
            F_z = self.L_z @ F_z
        return F_z

    def _F_XW(self, q, pi):
        '''Build a block between X states of F(q).'''
        F_XW = self._F_a(q, pi)
        return F_XW

    def _F_yy(self, q, pi, xi):
        '''Build a diagonal block for a y state of F(q).'''
        F_a = self._F_a(q, pi)
        F_z = self._F_z(q, xi)
        H_a = self._H_a(q)
        H_z = self._H_z(q)
        F_yy = (_utility.sparse.kron(F_a, H_z)
                + _utility.sparse.kron(H_a, F_z))
        return F_yy

    def _F_Xy(self, q, xi):
        '''Build a block to an X state from a y state of F(q).'''
        H_a = self._H_a(q)
        sigma_z = self._sigma_z(xi)
        F_Xy = _utility.sparse.kron(H_a, sigma_z)
        return F_Xy

    def _F_yw(self, q, xi):
        '''Build an off-diagonal block between y states of F(q).'''
        H_a = self._H_a(q)
        zeta_z = self.zeta_z
        sigma_z = self._sigma_z(xi)
        F_yw = _utility.sparse.kron(H_a, zeta_z @ sigma_z)
        return F_yw

    def _get_rate_a(self, which):
        '''Get the age-dependent rate `which` and make finite any
        infinite entries.'''
        param = getattr(self.model.parameters, which)
        rate = param.rate(self.a)
        return _utility.numerical.rate_make_finite(rate)

    def _get_rate_z(self, which):
        '''Get the time-since-entry-dependent rate `which` and make
        finite any infinite entries.'''
        param = getattr(self.model.parameters, which)
        rate = param.rate(self.z)
        return _utility.numerical.rate_make_finite(rate)

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        mu = self._get_rate_a('death')
        omega = self._get_rate_a('waning')
        rho = self._get_rate_z('progression')
        gamma = self._get_rate_z('recovery')
        F_XW = functools.partial(self._F_XW, q)
        F_yy = functools.partial(self._F_yy, q)
        F_Xy = functools.partial(self._F_Xy, q)
        F_yw = functools.partial(self._F_yw, q)
        F = _utility.sparse.bmat([
            [F_XW(- omega - mu), None, None, None, None],
            [F_XW(omega), F_XW(- mu), None, None, None],
            [None, None, F_yy(- mu, - rho), None, None],
            [None, None, F_yw(rho), F_yy(- mu, - gamma), None],
            [None, None, None, F_Xy(gamma), F_XW(- mu)]
        ])
        return F

    def _T_a(self, q):
        '''Build an age block of T(q).'''
        T_a = self._H_a(q)
        return T_a

    def _T_XX(self, q):
        '''Build a diagonal block for an X state of T(q).'''
        T_XX = self._T_a(q)
        return T_XX

    def _T_yX(self, q):
        '''Build a block to a y state from an X state of T(q).'''
        T_a = self._T_a(q)
        zeta_z = self.zeta_z
        T_yX = _utility.sparse.kron(T_a, zeta_z)
        return T_yX

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        T_XX = self._T_XX(q)
        T_yX = self._T_yX(q)
        Zeros_XW = self.Zeros['XW']
        Zeros_Xy = self.Zeros['Xy']
        Zeros_yw = self.Zeros['yw']
        T = _utility.sparse.bmat([
            [Zeros_XW,   None, Zeros_Xy,     None,     None],
            [    None, - T_XX,     None,     None,     None],
            [    None,   T_yX,     None,     None,     None],
            [    None,   None,     None, Zeros_yw,     None],
            [    None,   None,     None,     None, Zeros_XW]
        ])
        return T

    def _B_XW(self):
        '''Build a block between X states of B.'''
        b_a = self.b_a
        tau_X = self._tau_X()
        B_XW = b_a @ tau_X
        return B_XW

    def _B_Xy(self):
        '''Build a block to an X state from a y state of B.'''
        b_a = self.b_a
        tau_y = self._tau_y()
        B_Xy = b_a @ tau_y
        return B_Xy

    def _B(self):
        '''Build the birth matrix B.'''
        B_XW = self._B_XW()
        B_Xy = self._B_Xy()
        Zeros_XW = self.Zeros['XW']
        Zeros_yX = self.Zeros['yX']
        B = _utility.sparse.bmat([
            [    None, None, None, None, B_XW],
            [    B_XW, B_XW, B_Xy, B_Xy, None],
            [Zeros_yX, None, None, None, None],
            [Zeros_yX, None, None, None, None],
            [Zeros_XW, None, None, None, None]
        ])
        return B

    def _check_matrices(self, is_M_matrix=False):
        '''Check the solver matrices. Checking the M matrix requires
        finding the dominant eigenvalue, which is very slow and very
        memory intensive.'''
        super()._check_matrices(is_M_matrix=is_M_matrix)
