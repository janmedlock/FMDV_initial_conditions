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
        # pylint: disable-next=assignment-from-none
        idx_other = super()._build_index()
        if idx_other is not None:
            raise NotImplementedError
        states = self.states
        return pandas.CategoricalIndex(states, states,
                                       ordered=True, name='state')

    @functools.cached_property
    def _weights(self):
        '''Weights for the 'state' level.'''
        weights_other = super()._weights
        if weights_other is not None:
            raise NotImplementedError
        # Each 'state' has weight 1.
        return pandas.Series(1, index=self._index)

    def build_initial_conditions(self):
        '''Build the initial conditions for the 'state' level.'''
        # pylint: disable-next=invalid-name,assignment-from-none
        Y_other = super().build_initial_conditions()
        if Y_other is not None:
            raise NotImplementedError
        M = E = R = 0  # pylint: disable=invalid-name
        I = 0.01  # pylint: disable=invalid-name  # noqa: E741
        S = 1 - M - E - I - R  # pylint: disable=invalid-name
        idx_state = self._get_index_level('state')
        # pylint: disable-next=invalid-name
        Y_state = pandas.Series([M, S, E, I, R], index=idx_state)
        # Broadcast out to the full index `self._index`.
        ones = pandas.Series(1, index=self._index)
        return Y_state * ones
