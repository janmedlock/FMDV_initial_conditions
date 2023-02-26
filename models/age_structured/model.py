'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _solver
from .. import _model
from .._utility import numerical


class Model(_model.AgeDependent):
    '''Age-structured model.'''

    _Solver = _solver.Solver

    def __init__(self, a_step=0.001, a_max=25, **kwds):
        self.a_step = a_step
        self.a = numerical.build_t(0, a_max, self.a_step)
        super().__init__(**kwds)

    def _build_solution_index(self, states):
        '''Build the solution index.'''
        ages = pandas.Index(self.a, name='age')
        states_ages = pandas.MultiIndex.from_product((states, ages))
        return states_ages

    def _build_weights(self):
        '''Build weights for the state vector.'''
        return self.a_step

    def initial_conditions_from_unstructured(self, Y, *args, **kwds):
        '''Build initial conditions from the unstructured `Y`.'''
        n = self.stable_age_density(*args, **kwds)
        # [X * n for X in Y] stacked in one big vector.
        return numpy.kron(Y, n)

    def build_initial_conditions(self, *args, **kwds):
        '''Build the initial conditions.'''
        # Totals over age in each immune state.
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        Y = (M, S, E, I, R)
        return self.initial_conditions_from_unstructured(Y, *args, **kwds)
