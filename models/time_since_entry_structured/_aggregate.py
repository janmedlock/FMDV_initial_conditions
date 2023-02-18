'''`pandas` accessors to integrate over time since entry.'''

import numpy
import pandas


_ACCESSOR_NAME = 'time_since_entry'
_LEVEL = 'time_since_entry'


class _BaseAccessor:
    '''Common code for `SeriesAccessor()` and `DataFrameAccessor()`.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def _aggregate_one(obj, axis):
        '''Integrate one group over time since entry.'''
        z = obj.axes[axis].get_level_values(_LEVEL)
        if len(z) == 1:
            z_step = 1
        else:
            z_step = numpy.mean(numpy.diff(z))
        return obj.sum(axis=axis) * z_step

    def aggregate(self):
        '''Integrate over time since entry.'''
        # Operate on the last axis.
        axis = self._obj.ndim - 1
        # Group by all the levels on `axis` except `_LEVEL`.
        others = self._obj.axes[axis].names.difference({_LEVEL})
        grouper = self._obj.groupby(others, axis=axis)
        return grouper.apply(self._aggregate_one, axis)


@pandas.api.extensions.register_series_accessor(_ACCESSOR_NAME)
class SeriesAccessor(_BaseAccessor):
    '''API to integrate over time since entry.'''


@pandas.api.extensions.register_dataframe_accessor(_ACCESSOR_NAME)
class DataFrameAccessor(_BaseAccessor):
    '''API to integrate over time since entry.'''
