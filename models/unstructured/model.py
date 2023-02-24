'''Based on our FMDV work, this is an unstructured model.'''

from . import _solver
from .. import _model


class Model(_model.AgeIndependent):
    '''Unstructured model.'''

    _Solver = _solver.Solver

    def __init__(self, t_step=0.001, **kwds):
        self.t_step = t_step
        super().__init__(**kwds)

    def _build_solution_index(self, states):
        return states

    def _build_weights(self):
        '''Build weights for the state vector.'''
        return 1

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        return (M, S, E, I, R)
