'''`pandas` accessor to integrate over time since entry.'''

import numpy

from .._model import aggregate


class TimeSinceEntryAccessor(aggregate.OneDimAccessor):
    '''API to integrate over time since entry.'''

    _accessor_name = 'time_since_entry'
    _level = 'time_since_entry'

    @classmethod
    def _aggregate_one(cls, obj, axis):
        '''Integrate one group over time since entry.'''
        z = cls._get_level_values(obj, axis)
        if len(z) == 1:
            z_step = 1
        else:
            z_step = numpy.mean(numpy.diff(z))
        return obj.sum(axis=axis) * z_step


TimeSinceEntryAccessor._register()
