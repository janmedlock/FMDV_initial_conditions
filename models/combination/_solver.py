'''Solver.'''

from .. import _model


class Solver(_model.solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = True

    def __init__(self, model):
        self.a_step = self.z_step = self.t_step = model.z_step
        super().__init__(model)

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
