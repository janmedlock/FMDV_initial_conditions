'''Based on our FMDV work, this is an age– and
time-since-entry–structured model.'''

from . import _solver
from .. import (age_structured, time_since_entry_structured,
                unstructured, _model)
from .._utility import numerical


class Model(_model.Model, unstructured.Mixin, age_structured.Mixin,
            time_since_entry_structured.Mixin):
    '''Age– and time-since-entry–structured model.'''

    _Solver = _solver.Solver

    def __init__(self, step=0.001, a_max=25, z_max=3, **kwds):
        self.a_step = self.z_step = step
        self.a = numerical.build_t(0, a_max, self.a_step)
        self.z = numerical.build_t(0, z_max, self.z_step)
        super().__init__(**kwds)

    def _build_index(self):
        '''Build the solution index.'''
        idx_state_age = self._build_index_state_age()
        idx = self._extend_index_with_z(idx_state_age)
        return idx

    def _build_weights(self):
        '''Build weights for the state vector.'''
        weights_state_age = self._build_weights_state_age()
        weights = self._adjust_weights_for_z(weights_state_age)
        return weights

    def build_initial_conditions(self, how='all_in_first', *args, **kwds):
        '''Build the initial conditions.'''
        Y_state_age = self._build_initial_conditions_state_age(*args, **kwds)
        y = self._adjust_initial_conditions_for_z(Y_state_age, how=how)
        return y
