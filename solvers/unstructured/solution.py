'''Classes to hold solutions and points in state space.'''

import cycler
import matplotlib.pyplot
import numpy
import pandas

from .. import utility


def Solution(y, t=None, states=None):
    '''A solution.'''
    if states is not None:
        states = pandas.Index(states, name='state')
    if t is None:
        return pandas.Series(y, index=states)
    else:
        index = pandas.Index(t, name='time')
        return pandas.DataFrame(y, index=index, columns=states)


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

    @property
    def _dim_state(self):
        n_states = len(self.states)
        if n_states < 2:
            raise ValueError(f'{len(self.states)=}')
        elif n_states == 2:
            return 2
        else:  # n_states >= 3.
            return 3

    def _make_axes_state(self):
        '''Make state axes.'''
        dim_state = self._dim_state
        if dim_state == 2:
            projection = 'rectilinear'
        elif dim_state == 3:
            projection = '3d'
        else:
            raise ValueError(f'{dim_state=}')
        fig = matplotlib.pyplot.figure()
        axis_labels = dict(zip(('xlabel', 'ylabel', 'zlabel'), self.states))
        return fig.add_subplot(projection=projection,
                               **axis_labels)


@pandas.api.extensions.register_series_accessor('solution')
class _SolutionSeriesAccessor(_SolutionAccessorBase):
    '''API for a point in state space.'''

    def plot_state(self, ax=None, **kwds):
        '''Plot the point in state space.'''
        if ax is None:
            ax = self._make_axes_state()
        ax.scatter(*self._obj[:self._dim_state], **kwds)
        return ax


@pandas.api.extensions.register_dataframe_accessor('solution')
class _SolutionDataFrameAccessor(_SolutionAccessorBase):
    '''API for state vs. time.'''

    def _prop_cycler_solution(self):
        orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
        inner = orig[:len(self.states)]
        outer = cycler.cycler(linestyle=('solid', 'dotted', 'dashed'))
        return outer * inner

    def _make_axes_solution(self):
        fig = matplotlib.pyplot.figure()
        return fig.add_subplot(prop_cycle=self._prop_cycler_solution())

    def plot_solution(self, ax=None, **kwds):
        '''Plot the solution vs time.'''
        if ax is None:
            ax = self._make_axes_solution()
        return self._obj.plot(ax=ax, **kwds)

    def plot_state(self, ax=None, **kwds):
        '''Make a phase plot.'''
        if ax is None:
            ax = self._make_axes_state()
        cols = (col for (_, col) in self._obj.items())
        ax.plot(*cols, **kwds)
        return ax
