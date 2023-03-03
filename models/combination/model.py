'''Based on our FMDV work, this is an age– and
time-since-entry–structured model.'''

from . import _solver
from .. import (age_structured, time_since_entry_structured,
                unstructured, _model)


class Model(_model.Model,
            time_since_entry_structured.Mixin,
            age_structured.Mixin):
    '''Age– and time-since-entry–structured model.'''

    _Solver = _solver.Solver

    def __init__(self,
                 a_max=age_structured.Mixin.DEFAULT_A_MAX,
                 z_max=time_since_entry_structured.Mixin.DEFAULT_Z_MAX,
                 **kwds):
        self.a_max = a_max
        self.z_max = z_max
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

    def integral_over_a_and_z(self, obj, *args, **kwds):
        '''Integrate `obj` over 'age' and 'time_since_entry'.'''
        integrated_over_z = self.integral_over_z(obj,
                                                 *args, **kwds)
        integrated_over_z_and_a = self.integral_over_a(integrated_over_z,
                                                       *args, **kwds)
        return integrated_over_z_and_a

    integral_over_z_and_a = integral_over_a_and_z
