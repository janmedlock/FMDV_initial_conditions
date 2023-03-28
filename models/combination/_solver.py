'''Solver.'''

import functools

from .. import _model, _utility


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _sparse = True

    _jacobian_method_default = 'sparse_csc'

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

    def _I(self):
        '''Build the identity matrix.'''

    def _beta(self):
        '''Build the transmission rate vector beta.'''

    def _H(self, q):
        '''Build the time step matrix H(q).'''

    def _F(self, q):
        '''Build the transition matrix F(q).'''

    def _T(self, q):
        '''Build the transition matrix F(q).'''

    def _B(self):
        '''Build the birth matrix B.'''

    def _check_matrices(self):
        '''TODO: Remove me so that super()._check_matrices() runs.'''

    def _preconditioner(self):
        '''TODO: Remove me so that super()._preconditioner() runs.'''
