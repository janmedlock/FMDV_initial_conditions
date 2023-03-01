'''Plot solutions and points in state space.'''

import cycler
import matplotlib.pyplot


def population_size(obj):
    '''Population size.'''
    # Sum over the last axis.
    axis = obj.ndim - 1
    return obj.sum(axis=axis)


def _get_states(obj, states):
    '''Get the states to plot.'''
    if states is None:
        return obj.axes[-1]
    return states


def _state_make_axes(states):
    '''Make state axes.'''
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


def state(obj, states=None, ax=None, **kwds):
    '''Make a phase plot.'''
    states = _get_states(obj, states)
    if ax is None:
        ax = _state_make_axes(states)
    if obj.ndim == 1:
        ax.scatter(*obj[states], **kwds)
    elif obj.ndim == 2:
        # Not `col = obj[states]` because the result must iterate over
        # columns, not rows.
        cols = (obj[col] for col in states)
        ax.plot(*cols, **kwds)
    else:
        raise ValueError(f'{obj.ndim=}')
    return ax


def _solution_prop_cycler(states):
    '''Build a prop_cycler for the solution plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    inner = orig[:len(states)]
    outer = cycler.cycler(linestyle=('solid', 'dotted', 'dashed'))
    return outer * inner


def _solution_make_axes(states):
    '''Make solution axes.'''
    fig = matplotlib.pyplot.figure()
    prop_cycle = _solution_prop_cycler(states)
    return fig.add_subplot(prop_cycle=prop_cycle)


def solution(obj, states=None, ax=None, **kwds):
    '''Plot the solution vs time.'''
    states = _get_states(obj, states)
    if ax is None:
        ax = _solution_make_axes(states)
    return obj[states].plot(ax=ax, **kwds)
