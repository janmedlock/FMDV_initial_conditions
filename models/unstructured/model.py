'''Based on our FMDV work, this is an unstructured model.'''

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

    # The default time step `t_step`. A necessary condition for
    # nonnegative solutions is is that `t_step` must be less than 1 /
    # rate for all of the model transition rates. In particular,
    # `transmission_rate` and 1 / `progression_mean`, especially for
    # SAT1, are just a bit less than 1000.
    _t_step_default = 1e-3

    def __init__(self, t_step=_t_step_default, **kwds):
        super().__init__(t_step, **kwds)

    def _build_index(self):
        '''Build the 'state' level `pandas.Index()` for solutions.'''
        # Use `pandas.CategoricalIndex()` to preserve the order of the
        # states.
        idx_other = super()._build_index()
        states = self.states
        idx = pandas.CategoricalIndex(states, states,
                                      ordered=True, name='state')
        return idx

    def _build_weights(self):
        '''Build weights for the 'state' level.'''
        weights_other = super()._build_weights()
        # Each 'state' has weight 1.
        weights = pandas.Series(1, index=self._index)
        return weights

    def build_initial_conditions(self):
        '''Build the initial conditions for the 'state' level.'''
        Y_other = super().build_initial_conditions()
        if Y_other is not None:
            clsname = self.__class__.__qualname__
            msg = f'{clsname} should be the last inherited class.'
            raise NotImplementedError(msg)
        M = E = R = 0
        I = 0.01
        S = 1 - M - E - I - R
        idx_state = self._get_index_level('state')
        Y_state = pandas.Series([M, S, E, I, R], index=idx_state)
        # Broadcast out to the full index `self._index`.
        ones = pandas.Series(1, index=self._index)
        Y = Y_state * ones
        return Y
