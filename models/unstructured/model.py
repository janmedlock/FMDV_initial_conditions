'''Based on our FMDV work, this is an unstructured model.'''

import numpy
import pandas

from . import _solver
from .. import _model


class Model(_model.AgeIndependent):
    '''Unstructured model.'''

    _Solver = _solver.Solver

    def __init__(self, t_step=0.001, **kwds):
        self.t_step = t_step
        super().__init__(**kwds)

    def _build_solution_index(self):
        '''Build a `pandas.Index()` for solutions.'''
        # Use `pandas.CategoricalIndex()` to preserve the order of the
        # states.
        states = self.states
        idx = pandas.CategoricalIndex(states, states,
                                      ordered=True, name='state')
        return idx

    def _build_weights(self):
        '''Build weights for the state vector.'''
        n = len(self.states)
        weights = numpy.ones(n)
        return weights

    def build_initial_conditions(self):
        '''Build the initial conditions.'''
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        Y = numpy.array([M, S, E, I, R])
        return Y
