'''Based on our FMDV work, this is an age– and
time-since-entry–structured model.'''

import numpy

from . import solver
from .. import age_structured, time_since_entry_structured


# Inheriting from `time_since_entry_structured.Model` first makes
# 'time_since_entry' the last variable in `Model._index`.
class Model(time_since_entry_structured.Model, age_structured.Model):
    '''Age– and time-since-entry–structured model.'''

    # Because the the 'maternal_immunity' class starts at birth ('age'
    # = 0), 'age' and 'time_since_entry' are identical, so we can drop
    # 'time_since_entry'.
    states_with_z = time_since_entry_structured.Model.states_with_z.copy()
    states_with_z.remove('maternal_immunity')

    _Solver = solver.Solver

    def integral_over_a_and_z(self, obj, *args, **kwds):
        '''Integrate `obj` over 'age' and 'time_since_entry'.'''
        integrated_over_z = self.integral_over_z(obj, *args, **kwds)
        return self.integral_over_a(integrated_over_z,
                                    *args, **kwds)

    integral_over_z_and_a = integral_over_a_and_z

    def _fix_maternal_immunity(self, y, how):
        '''Get scaled survivals.'''
        if how == 'survival':
            waiting_time = getattr(self.parameters,
                                   self._waiting_times_z['maternal_immunity'])
            shape = waiting_time.survival(self.a)
            # Scale to integrate to 1.
            shape /= self.integral_over_a(shape)
        elif how == 'all_in_first':
            shape = numpy.hstack(
                [1 / self.a_step, numpy.zeros(len(self.a) - 1)]
            )
        else:
            raise ValueError(f'Unknown {how=}!')
        total = self.integral_over_a(y['maternal_immunity'].to_numpy())
        y['maternal_immunity'] = shape * total

    def build_initial_conditions(self, how='survival', **kwds):
        '''Build the initial conditions.'''
        y = super().build_initial_conditions(how=how, **kwds)
        self._fix_maternal_immunity(y, how)
        return y
