'''`pandas` accessor to integrate over age.'''

import numpy

from .._model import aggregate


class AgeAccessor(aggregate.OneDimAccessor):
    '''API to integrate over age.'''

    _accessor_name = 'age'
    _level = 'age'

    @classmethod
    def _aggregate_one(cls, obj, axis):
        '''Integrate one group over age.'''
        a = cls._get_level_values(obj, axis)
        a_step = numpy.mean(numpy.diff(a))
        return obj.sum(axis=axis) * a_step


AgeAccessor._register()
