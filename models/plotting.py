'''Plot solutions and points in state space.'''

import functools

import matplotlib.pyplot
import matplotlib.rcsetup


_linestyles = matplotlib.rcsetup.cycler(
    linestyle=['solid', 'dotted', 'dashed']
)

_markers = matplotlib.rcsetup.cycler(
    marker=['o', '^', 's', '*', '+', 'x']
)


def _cycler_inner_product(*cyclers):
    '''Get the inner product of the cyclers, reducing their length to
    that of the smallest one.'''
    # Make them all the same length.
    stop = min(map(len, cyclers))
    cyclers_ = map(lambda x: x[:stop], cyclers)
    # Sum them.
    return functools.reduce(lambda x, y: x + y, cyclers_)


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


def _state_prop_cycle():
    '''Build a prop_cycle for the state plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    prop_cycle = _cycler_inner_product(orig, _markers, _linestyles)
    return prop_cycle


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


def _solution_prop_cycle(states):
    '''Build a prop_cycle for the solution plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    inner = orig[:len(states)]
    outer = _linestyles
    prop_cycle = outer * inner
    return prop_cycle


def _solution_make_axes(states):
    '''Make solution axes.'''
    fig = matplotlib.pyplot.figure()
    prop_cycle = _solution_prop_cycle(states)
    return fig.add_subplot(prop_cycle=prop_cycle)


def solution(obj, states=None, ax=None, **kwds):
    '''Plot the solution vs time.'''
    states = _get_states(obj, states)
    if ax is None:
        ax = _solution_make_axes(states)
    return obj[states].plot(ax=ax, **kwds)


def _eigvals_prop_cycle():
    '''Build a prop_cycle for the eigenvalue plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    prop_cycle = _cycler_inner_product(orig, _markers)
    return prop_cycle


def _eigvals_make_axes():
    fig = matplotlib.pyplot.figure()
    prop_cycle = _eigvals_prop_cycle()
    axis_labels = dict(xlabel=r'$\Re(\lambda)$',
                       ylabel=r'$\Im(\lambda)$')
    return fig.add_subplot(prop_cycle=prop_cycle,
                           **axis_labels)


def eigvals(eigs, ax=None, legend=False, **kwds):
    if ax is None:
        ax = _eigvals_make_axes()
    ax.plot(eigs.real, eigs.imag, linestyle='', **kwds)
    if legend:
        ax.legend()
    return ax