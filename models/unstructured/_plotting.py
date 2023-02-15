'''Classes to plot solutions and points in state space.'''

import cycler
import matplotlib.pyplot
import pandas


class _SolutionAccessorBase:
    '''Common code for `_SolutionSeriesAccessor()` and
    `_SolutionDataFrameAccessor()`.'''

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def states(self):
        '''State coordinates.'''
        return self._obj.axes[-1]

    @property
    def population_size(self):
        '''Population size.'''
        # Sum over the last axis.
        axis = self._obj.ndim - 1
        return self._obj.sum(axis=axis)

    def _get_states(self, states):
        if states is None:
            return self.states
        return states

    def _make_axes_state(self, states):
        '''Make state axes.'''
        states = self._get_states(states)
        if len(states) == 2:
            projection = 'rectilinear'
        elif len(states) == 3:
            projection = '3d'
        else:
            raise ValueError(f'{len(states)=}')
        fig = matplotlib.pyplot.figure()
        axis_labels = dict(zip(('xlabel', 'ylabel', 'zlabel'), states))
        return fig.add_subplot(projection=projection,
                               **axis_labels)


@pandas.api.extensions.register_series_accessor('solution')
class _SolutionSeriesAccessor(_SolutionAccessorBase):
    '''API for a point in state space.'''

    def plot_state(self, states=None, ax=None, **kwds):
        '''Plot the point in state space.'''
        states = self._get_states(states)
        if ax is None:
            ax = self._make_axes_state(states)
        ax.scatter(*self._obj[states], **kwds)
        return ax


@pandas.api.extensions.register_dataframe_accessor('solution')
class _SolutionDataFrameAccessor(_SolutionAccessorBase):
    '''API for state vs. time.'''

    def _prop_cycler_solution(self, states):
        states = self._get_states(states)
        orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
        inner = orig[:len(states)]
        outer = cycler.cycler(linestyle=('solid', 'dotted', 'dashed'))
        return outer * inner

    def _make_axes_solution(self, states):
        fig = matplotlib.pyplot.figure()
        return fig.add_subplot(prop_cycle=self._prop_cycler_solution(states))

    def plot_solution(self, states=None, ax=None, **kwds):
        '''Plot the solution vs time.'''
        states = self._get_states(states)
        if ax is None:
            ax = self._make_axes_solution(states)
        return self._obj[states].plot(ax=ax, **kwds)

    def plot_state(self, states=None, ax=None, **kwds):
        '''Make a phase plot.'''
        states = self._get_states(states)
        if ax is None:
            ax = self._make_axes_state(states)
        # Not `col = self._obj[states]` because the result must
        # iterate over columns, not rows.
        cols = (self._obj[col] for col in states)
        ax.plot(*cols, **kwds)
        return ax
