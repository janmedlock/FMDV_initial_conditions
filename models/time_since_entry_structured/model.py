'''Based on our FMDV work, this is a time-since-entry-structured model.'''

import numpy
import pandas

from . import _solver
from .. import unstructured, _model, _utility


class Model(unstructured.Model):
    '''Time-since-entry-structured model.'''

    states_with_z = ['maternal_immunity', 'exposed', 'infectious']

    _Solver = _solver.Solver

    # The default maximum time since entry `z_max`.
    # This was chosen to ensure that the survival for all of the
    # time-since-entry-dependent model processes at `z_max` was
    # small. Those processes are 'waning', 'progression', and
    # 'recovery': 'waning' is by far the slowest, i.e. the one that
    # determines where `z_max` is large enough. See the method
    # `._check_survivals_at_zmax()` at the end of this class. At
    # `z_max` = 3, the survival for waning is less than 1.1e-4 and the
    # survivals for the other processes are less than 1e-270.
    _z_max_default = 3

    def __init__(self, z_max=_z_max_default, **kwds):
        assert z_max > 0
        self.z_max = z_max
        super().__init__(**kwds)

    def _init_post(self):
        '''Final initialization.'''
        assert self.z_step > 0
        super()._init_post()

    @property
    def z_step(self):
        return self._Solver._get_z_step(self.t_step)

    def _extend_index(self, idx_other):
        '''Extend `idx_other` with the 'time-since-entry' level.'''
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

    def _build_index(self):
        '''Extend the `pandas.Index()` for solutions with the
        'time-since-entry' level.'''
        self.z = _utility.numerical.build_t(0, self.z_max, self.z_step)
        idx_other = super()._build_index()
        idx = self._extend_index(idx_other)
        return idx

    def _build_weights(self):
        '''Adjust the weights for the 'time-since-entry' level.'''
        weights_other = super()._build_weights()
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

    def _survivals_scaled(self):
        '''Get scaled survivals.'''
        idx_state = self._get_index_level('state')
        idx_state_z = self._extend_index(idx_state)
        n_z = pandas.Series(1, index=idx_state_z)
        n_z['maternal_immunity'] = self._survival_scaled(
            self.parameters.waning
        )
        n_z['exposed'] = self._survival_scaled(self.parameters.progression)
        n_z['infectious'] = self._survival_scaled(self.parameters.recovery)
        return n_z

    def _all_in_first(self):
        '''Get a vector with all the density in the first time since
        entry.'''
        idx_state = self._get_index_level('state')
        idx_state_z = self._extend_index(idx_state)
        n_z = pandas.Series(1, index=idx_state_z)
        idx_z = self._get_index_level('time_since_entry')
        n_with_z = pandas.Series(0, index=idx_z)
        n_with_z.iloc[0] = 1 / self.z_step
        for state in self.states_with_z:
            n_z[state] = n_with_z
        return n_z

    def build_initial_conditions(self, how='survival', **kwds):
        '''Adjust the initial conditions for the 'time-since-entry' level.'''
        Y_other = super().build_initial_conditions(**kwds)
        idx_state = self._get_index_level('state')
        idx_state_z = self._extend_index(idx_state)
        n_z = pandas.Series(1, index=idx_state_z)
        if how == 'survival':
            # Use survivals to get density within states.
            n_z = self._survivals_scaled()
        elif how == 'all_in_first':
            # For states with 'time_since_entry', put all of the
            # density in the first 'time_since_entry'.
            n_z = self._all_in_first()
        else:
            raise ValueError(f'Unknown {how=}!')
        y = Y_other * n_z
        if len(y.index.names) > 2:
            # Fix the index of `y`.
            # Move 'time_since_entry' from the 2nd position to the
            # last position.
            order = [0, *range(2, len(y.index.names)), 1]
            y = y.reorder_levels(order)
            # Fix index dtypes.
            dtypes = Y_other.index.dtypes
            dfr = y.index.to_frame().astype(dtypes)
            idx = pandas.MultiIndex.from_frame(dfr)
            y = y.set_axis(idx)
            y.sort_index(inplace=True)
        return y

    @classmethod
    def _check_survivals_at_z_max(cls, **kwds):
        '''Get the survivals at `z_max` to check whether `z_max` is
        large enough.'''
        self = cls(**kwds)
        survivals_scaled = self._survivals_scaled() \
                               .loc[self.states_with_z]
        z = self.z
        survivals = survivals_scaled / survivals_scaled.loc[:, z[0]]
        return survivals.loc[:, z[-1]]
