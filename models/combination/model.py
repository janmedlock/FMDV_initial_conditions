'''Based on our FMDV work, this is an age– and
time-since-entry–structured model.'''

from . import _solver
from .. import age_structured, time_since_entry_structured


# Inheriting from `time_since_entry_structured.Model` first makes
# 'time_since_entry' the last variable in `Model._index`.
class Model(time_since_entry_structured.Model,
            age_structured.Model):
    '''Age– and time-since-entry–structured model.'''

    _Solver = _solver.Solver

    def integral_over_a_and_z(self, obj, *args, **kwds):
        '''Integrate `obj` over 'age' and 'time_since_entry'.'''
        integrated_over_z = self.integral_over_z(obj, *args, **kwds)
        integrated_over_z_and_a = self.integral_over_a(integrated_over_z,
                                                       *args, **kwds)
        return integrated_over_z_and_a

    integral_over_z_and_a = integral_over_a_and_z
