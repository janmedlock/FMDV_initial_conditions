'''Based on our FMDV work, this is a time-since-entry-structured model.'''

import numpy
import pandas

from . import _solver
from .. import _model
from .. import unstructured
from .._utility import numerical


class Model(_model.AgeIndependent):
    '''Time-since-entry-structured model.'''

    _Solver = _solver.Solver

    states_with_z = ['maternal_immunity', 'exposed', 'infectious']

    def __init__(self, z_step=0.001, z_max=3, **kwds):
        self.z_step = z_step
        self.z = numerical.build_t(0, z_max, self.z_step)
        super().__init__(**kwds)

    def _build_solution_index(self):
        '''Build the solution index.'''
        states = unstructured.Model._build_solution_index(self)
        # Build a `pandas.DataFrame()` with columns 'state' and
        # 'time_since_entry', then convert to a `pandas.MultiIndex()`.
        z = pandas.Series(self.z, name='time_since_entry')
        no_z = pandas.Series([numpy.NaN], name=z.name)
        blocks = []
        for state in states:
            z_vals = z if state in self.states_with_z else no_z
            s_vals = pandas.Series([state] * len(z_vals), name=states.name)
            blocks.append(pandas.concat([s_vals, z_vals], axis='columns'))
        dfr = pandas.concat(blocks)
        idx = pandas.MultiIndex.from_frame(dfr)
        return idx

    def _build_weights(self):
        '''Build weights for the state vector.'''
        weights_state = unstructured.Model._build_weights(self)
        K = len(self.z)
        z = self.z_step * numpy.ones(K)
        no_z = 1
        blocks = []
        for (state, s_val) in zip(self.states, weights_state):
            z_vals = z if state in self.states_with_z else no_z
            blocks.append(s_val * z_vals)
        weights = numpy.hstack(blocks)
        return weights

    def build_initial_conditions(self):
        '''Build the initial conditions.'''
        Y = unstructured.Model.build_initial_conditions(self)
        [M, S, E, I, R] = Y
        # Put all of the density in the first time since entry.
        K = len(self.z)
        n = numpy.hstack([1 / self.z_step, numpy.zeros(K - 1)])
        [m, e, i] = numpy.outer([M, E, I], n)
        y = numpy.hstack([m, S, e, i, R])
        return y

    def _survival_scaled(self, waiting_time):
        '''`waiting_time.survival(z)` scaled to integrate to 1.'''
        survival = waiting_time.survival(self.z)
        # Scale to integrate to 1.
        total = survival.sum() * self.z_step
        return survival / total

    def initial_conditions_from_unstructured(self, Y):
        '''Build initial conditions from the unstructured `Y`.'''
        [M, S, E, I, R] = Y
        # Use survivals to spread density within states.
        m = M * self._survival_scaled(self.waning)
        e = E * self._survival_scaled(self.progression)
        i = I * self._survival_scaled(self.recovery)
        y = numpy.hstack([m, S, e, i, R])
        return y
