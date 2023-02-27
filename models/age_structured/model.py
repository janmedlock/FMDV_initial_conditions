'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _solver
from .. import _model
from .. import unstructured
from .._utility import numerical


class Model(_model.AgeDependent):
    '''Age-structured model.'''

    _Solver = _solver.Solver

    def __init__(self, a_step=0.001, a_max=25, **kwds):
        self.a_step = a_step
        self.a = numerical.build_t(0, a_max, self.a_step)
        super().__init__(**kwds)

    def _build_solution_index(self):
        '''Build a `pandas.Index()` for solutions.'''
        states = unstructured.Model._build_solution_index(self)
        ages = pandas.Index(self.a, name='age')
        idx = pandas.MultiIndex.from_product((states, ages))
        return idx

    def _build_weights(self):
        '''Build weights for the state vector.'''
        weights_state = unstructured.Model._build_weights(self)
        J = len(self.a)
        weights_age = self.a_step * numpy.ones(J)
        weights = numpy.kron(weights_state, weights_age)
        return weights

    def build_initial_conditions_from_unstructured(self, Y, *args, **kwds):
        '''Build initial conditions from the unstructured `Y`.'''
        n = self.stable_age_density(*args, **kwds)
        y = numpy.kron(Y, n)
        return y

    def build_initial_conditions(self, *args, **kwds):
        '''Build the initial conditions.'''
        Y = unstructured.Model.build_initial_conditions(self)
        y = self.build_initial_conditions_from_unstructured(Y, *args, **kwds)
        return y
