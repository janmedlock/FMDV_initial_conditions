'''Based on our FMDV work, this is a time-since-entry-structured model.'''

import numpy
import pandas

from . import _solver
from .. import unstructured, _model
from .._utility import numerical


class Mixin:
    '''Attributes for models that have a time-since-entry variable.'''

    states_with_z = ['maternal_immunity', 'exposed', 'infectious']

    def _extend_index_with_z(self, idx_other):
        '''Append a new level for 'time since entry'.'''
        # Build a `pandas.DataFrame()` with the columns from
        # `idx_other` and 'time_since_entry', then convert to a
        # `pandas.MultiIndex()`.
        dfr_other = idx_other.to_frame()
        z_with_z = pandas.Series(self.z, name='time_since_entry')
        z_without_z = pandas.Series([numpy.NaN], name=z_with_z.name)
        blocks = []
        for state in self.states:
            other = dfr_other[dfr_other['state'] == state]
            z = z_with_z if state in self.states_with_z else z_without_z
            blocks.append(other.merge(z, how='cross'))
        dfr = pandas.concat(blocks)
        idx = pandas.MultiIndex.from_frame(dfr)
        return idx

    def _adjust_weights_for_z(self, weights_other):
        '''Adjust weights for 'time since entry'.'''
        # For states with 'time_since_entry', each 'time_since_entry'
        # has weight `self.z_step`.  For states without
        # 'time_since_entry', each has weight 1.
        idx_state = self._get_index_level('state')
        weights_z = pandas.Series(1, index=idx_state)
        weights_z[self.states_with_z] = self.z_step
        weights = weights_other * weights_z
        return weights

    def _integral_over_z_group(self, obj, axis):
        '''Integrate one group over time since entry.'''
        z = _model.integral.get_level_values(obj, axis, 'time_since_entry')
        if len(z) == 1:
            z_step = 1
        else:
            assert len(z) == len(self.z)
            z_step = self.z_step
        return obj.sum(axis=axis) * z_step

    def integral_over_z(self, obj, *args, **kwds):
        '''Integrate `obj` over 'time_since_entry'.'''
        if isinstance(obj, numpy.ndarray):
            assert len(obj) == len(self.z)
            return obj.sum(*args, **kwds) * self.z_step
        elif isinstance(obj, (pandas.Series, pandas.DataFrame)):
            return _model.integral.integral(obj, 'time_since_entry',
                                            self._integral_over_z_group)
        else:
            raise NotImplementedError

    def _survival_scaled(self, waiting_time):
        '''`waiting_time.survival(z)` scaled to integrate to 1.'''
        survival = waiting_time.survival(self.z)
        # Scale to integrate to 1.
        total = self.integral_over_z(survival)
        return survival / total

    def _adjust_initial_conditions_for_z(self, Y_other, how='all_in_first'):
        '''Adjust initial conditions for 'time since entry'.'''
        idx_state = self._get_index_level('state')
        idx_state_z = self._extend_index_with_z(idx_state)
        n_z = pandas.Series(1, index=idx_state_z)
        if how == 'all_in_first':
            # For states with 'time_since_entry', put all of the
            # density in the first 'time_since_entry'.
            idx_z = self._get_index_level('time_since_entry')
            n_with_z = pandas.Series(0, index=idx_z)
            n_with_z.iloc[0] = 1 / self.z_step
            for state in self.states_with_z:
                n_z[state] = n_with_z
        elif how == 'survival':
            # Use survivals to get density within states.
            n_z['maternal_immunity'] = self._survival_scaled(self.waning)
            n_z['exposed'] = self._survival_scaled(self.progression)
            n_z['infectious'] = self._survival_scaled(self.recovery)
        else:
            raise ValueError(f'Unknown {how=}!')
        y = Y_other * n_z
        if len(y.index.names) > 2:
            # Fix the index of y.
            # Move 'time_since_entry' from the 2nd position to the
            # end.
            order = [0, *range(2, len(y.index.names)), 1]
            y = y.reorder_levels(order)
            # Fix index dtypes.
            dtypes = Y_other.index.dtypes
            dfr = y.index.to_frame().astype(dtypes)
            idx = pandas.MultiIndex.from_frame(dfr)
            y = y.set_axis(idx)
            y.sort_index(inplace=True)
        return y


class Model(_model.ModelAgeIndependent, unstructured.Mixin, Mixin):
    '''Time-since-entry-structured model.'''

    _Solver = _solver.Solver

    def __init__(self, z_step=0.001, z_max=3, **kwds):
        self.z_step = z_step
        self.z = numerical.build_t(0, z_max, self.z_step)
        super().__init__(**kwds)

    def _build_index(self):
        '''Build a `pandas.Index()` for solutions.'''
        idx_state = self._build_index_state()
        idx = self._extend_index_with_z(idx_state)
        return idx

    def _build_weights(self):
        '''Build weights for the state vector.'''
        weights_state = self._build_weights_state()
        weights = self._adjust_weights_for_z(weights_state)
        return weights

    def build_initial_conditions(self, how='all_in_first'):
        '''Build the initial conditions.'''
        Y_state = self._build_initial_conditions_state()
        y = self._adjust_initial_conditions_for_z(Y_state, how=how)
        return y
