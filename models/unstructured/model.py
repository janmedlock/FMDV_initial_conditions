'''Based on our FMDV work, this is an unstructured model.'''

import functools

import pandas

from . import _solver
from .. import parameters, _model


class Model(_model.model.Model):
    '''Unstructured model.'''

    states = ['maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered']

    # This determines whether offspring are born with maternal
    # immunity.
    states_with_antibodies = ['recovered']

    _Parameters = parameters.ModelParametersAgeIndependent

    _Solver = _solver.Solver

    def _build_index(self):
        '''Build the 'state' level `pandas.Index()` for solutions.'''
        # Use `pandas.CategoricalIndex()` to preserve the order of the
        # states.
        idx_other = super()._build_index()
        if idx_other is not None:
            raise NotImplementedError
        states = self.states
        idx = pandas.CategoricalIndex(states, states,
                                      ordered=True, name='state')
        return idx

    @functools.cached_property
    def _weights(self):
        '''Weights for the 'state' level.'''
        weights_other = super()._weights
        if weights_other is not None:
            raise NotImplementedError
        # Each 'state' has weight 1.
        weights = pandas.Series(1, index=self._index)
        return weights

    def build_initial_conditions(self):
        '''Build the initial conditions for the 'state' level.'''
        Y_other = super().build_initial_conditions()
        if Y_other is not None:
            raise NotImplementedError
        M = E = R = 0
        I = 0.01
        S = 1 - M - E - I - R
        idx_state = self._get_index_level('state')
        Y_state = pandas.Series([M, S, E, I, R], index=idx_state)
        # Broadcast out to the full index `self._index`.
        ones = pandas.Series(1, index=self._index)
        Y = Y_state * ones
        return Y
