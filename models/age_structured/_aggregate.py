'''`pandas` accessors to integrate over age.'''

import numpy
import pandas


_ACCESSOR_NAME = 'age'
_LEVEL = 'age'


class _BaseAccessor:
    '''Common code for `SeriesAccessor()` and `DataFrameAccessor()`.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def _aggregate_one(obj, axis):
        '''Integrate one group over age.'''
        a = obj.axes[axis].get_level_values(_LEVEL)
        a_step = numpy.mean(numpy.diff(a))
        return obj.sum(axis=axis) * a_step

    def aggregate(self):
        '''Integrate over age.'''
        axis = self._obj.ndim - 1
        # Group by all the levels on `axis` except `_LEVEL`.
        others = self._obj.axes[axis].names.difference({_LEVEL})
        grouper = self._obj.groupby(others, axis=axis)
        return grouper.apply(self._aggregate_one, axis)


@pandas.api.extensions.register_series_accessor(_ACCESSOR_NAME)
class SeriesAccessor(_BaseAccessor):
    '''API to integrate over age.'''


@pandas.api.extensions.register_dataframe_accessor(_ACCESSOR_NAME)
class DataFrameAccessor(_BaseAccessor):
    '''API to integrate over age.'''
