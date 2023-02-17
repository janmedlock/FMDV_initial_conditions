'''Classes to integrate over age.'''

import numpy
import pandas


_ACCESSOR_NAME = 'age'
_LEVEL_NAME = 'age'


class _AggregateAccessorBase:
    '''Common code for `_AggregateSeriesAccessor()` and
    `_AggregateDataFrameAccessor()`.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def _get_axis(obj, axis):
        if axis in {0, 'index'}:
            return obj.index
        elif axis in {1, 'columns'}:
            return obj.columns
        raise ValueError

    @classmethod
    def _aggregate_one(cls, obj, axis):
        axis_ = cls._get_axis(obj, axis)
        ages = axis_.get_level_values(_LEVEL_NAME)
        age_step = numpy.mean(numpy.diff(ages))
        return obj.sum(axis=axis) * age_step

    def _aggregate(self, axis):
        axis_ = self._get_axis(self._obj, axis)
        others = axis_.names.difference({_LEVEL_NAME})
        grouper = self._obj.groupby(others, axis=axis)
        return grouper.apply(self._aggregate_one, axis)


@pandas.api.extensions.register_series_accessor(_ACCESSOR_NAME)
class _AggregateSeriesAccessor(_AggregateAccessorBase):
    '''API to integrate over age.'''

    def aggregate(self):
        '''Integrate over age.'''
        return self._aggregate('index')


@pandas.api.extensions.register_dataframe_accessor(_ACCESSOR_NAME)
class _AggregatenDataFrameAccessor(_AggregateAccessorBase):
    '''API to integrate over age.'''

    def aggregate(self):
        '''Integrate over age.'''
        return self._aggregate('columns')
