'''Classes to hold solutions and points in state space.'''

import matplotlib.pyplot
import numpy
import pandas

from . import utility


class _StateBase:
    def __init__(self, data):
        self._data = data

    @staticmethod
    def _index_states(states):
        return pandas.Index(states, name='state')

    def __repr__(self):
        return self._data.__repr__()

    @property
    def y(self):
        '''State values.'''
        return self._data.to_numpy()

    @property
    def states(self):
        '''State names.'''
        return self._data.axes[-1]

    @property
    def population_size(self):
        '''Population size.'''
        axis = self._data.ndim - 1  # Sum over last axis.
        return self._data.sum(axis=axis)

    def _get_state_axes(self):
        if len(self.states) == 2:
            projection = 'rectilinear'
        elif len(self.states) == 3:
            projection = '3d'
        else:
            raise ValueError(
                f'State dimension is {len(self._data.states)}, '
                'but only 2 and 3 are supported!')
        fig = matplotlib.pyplot.figure()
        labels = dict(zip(('xlabel', 'ylabel', 'zlabel'), self.states))
        return fig.add_subplot(projection=projection, **labels)


class State(_StateBase):
    '''Model state coordinates.'''
    def __init__(self, y, states=None):
        data = pandas.Series(y, index=states)
        super().__init__(data)

    def plot(self, ax=None, **kwds):
        '''Plot the point in state space.'''
        if ax is None:
            ax = self._get_state_axes()
        ax.scatter(*self.y, **kwds)
        return ax


class Solution(_StateBase):
    '''Model solution.'''
    def __init__(self, t, y, states=None):
        index = pandas.Index(t, name='$t$')
        data = pandas.DataFrame(y, columns=states, index=index)
        super().__init__(data)

    @property
    def t(self):
        '''Time values.'''
        return self._data.index.to_numpy()

    def interp(self, t):
        '''Interpolate to `t`.'''
        return utility.interp(t, self.t, self.y)

    def distance(self, t_0, t_1):
        '''Distance between solutions at `time_0` and `time_1`.'''
        (y_0, y_1) = self.interp([t_0, t_1])
        return numpy.linalg.norm(y_0 - y_1)

    def is_periodic(self, period, tol=1e-8):
        '''Whether the tail of the solution is periodic with period
        `period`.'''
        t_1 = self.t[-1]
        t_0 = t_1 - period
        return self.distance(t_0, t_1) < tol

    def plot(self, **kwds):
        '''Plot the solution.'''
        return self._data.plot(**kwds)

    def plot_phase(self, ax=None, **kwds):
        '''Make a phase plot.'''
        if ax is None:
            ax = self._get_state_axes()
        ax.plot(*self.y.T, **kwds)
        return ax
