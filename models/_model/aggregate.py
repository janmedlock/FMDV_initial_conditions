'''`pandas` accessors to integrate over levels.'''

import abc

import pandas


class BaseAccessor:
    '''Base accessor to integrate over levels.'''

    @property
    @abc.abstractmethod
    def _accessor_name(self):
        '''The name of the accessor.'''

    @classmethod
    @abc.abstractmethod
    def _aggregate(cls, obj):
        '''Integrate over levels of `obj`.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def aggregate(self):
        '''Integrate over levels.'''
        return self._aggregate(self._obj)

    @classmethod
    def _register(cls):
        '''Register the accessor with `pandas`.'''
        pd_api = pandas.api.extensions
        for which in ('series', 'dataframe'):
            reg_fcn = getattr(pd_api, f'register_{which}_accessor')
            reg_fcn(cls._accessor_name)(cls)


class OneDimAccessor(BaseAccessor):
    '''API to integrate over a level.'''

    @property
    @abc.abstractmethod
    def _level(self):
        '''The level to be integrated over.'''

    @classmethod
    @abc.abstractmethod
    def _aggregate_one(cls, obj, axis):
        '''Integrate one group.'''

    @classmethod
    def _get_level_values(cls, obj, axis):
        idx = obj.axes[axis]
        vals = idx.get_level_values(cls._level)
        return vals

    @classmethod
    def _aggregate(cls, obj):
        '''Integrate `obj`.'''
        # Operate on the last axis.
        axis = obj.ndim - 1
        # Group by all the levels on `axis` except `_level`.
        others = obj.axes[axis].names.difference({cls._level})
        grouper = obj.groupby(others, axis=axis, dropna=False)
        agg = grouper.apply(cls._aggregate_one, axis) \
                     .dropna()
        return agg
