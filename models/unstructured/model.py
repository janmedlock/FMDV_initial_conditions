'''Based on our FMDV work, this is an unstructured model.'''

import pandas

from . import _solver
from .. import parameters, _model, _utility


class Model(parameters.AgeIndependent,
            _model.model.Model):
    '''Unstructured model.'''

    _Solver = _solver.Solver

    def _build_index(self):
        '''Build the 'state' level `pandas.Index()` for solutions.'''
        # Use `pandas.CategoricalIndex()` to preserve the order of the
        # states.
        states = self.states
        idx = pandas.CategoricalIndex(states, states,
                                      ordered=True, name='state')
        return idx

    def _build_weights(self):
        '''Build weights for the 'state' level.'''
        # Each 'state' has weight 1.
        weights = pandas.Series(1, index=self._index)
        return weights

    def build_initial_conditions(self):
        '''Build the initial conditions for the 'state' level.'''
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        idx_state = self._get_index_level('state')
        Y_state = pandas.Series([M, S, E, I, R], index=idx_state)
        # Broadcast out to the full index `self._index`.
        ones = pandas.Series(1, index=self._index)
        Y = Y_state * ones
        return Y
